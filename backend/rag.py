from asyncio.log import logger
from html import parser

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from domains import DomainNames
import os
from collections import defaultdict
from llm import load_model
from Logger import Logging
import re
import uuid
from datetime import datetime
from dateutil.relativedelta import relativedelta  # pip install python-dateutil

# ── HYBRID SEARCH: new imports ───────────────────────────────────────────────
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
# ─────────────────────────────────────────────────────────────────────────────


class MultiDomainRAG:
    
    def __init__(self):
        self.pdf_directory = "pdfs/"
        self.documents_by_domain = defaultdict(list)
        self.document_metadata = defaultdict(list)
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.retrievers = {}
        self.vectorstores = {}
        tokenizer, model = load_model()
        self.model = model
        self.bm5Retrievers = {}
        self.prompt_domain = ''

        # ── HYBRID SEARCH: new attributes ────────────────────────────────────
        # bm25_indexes  : one BM25Okapi object per domain, built from the same
        #                 chunks that go into FAISS.
        # bm25_chunks   : the actual Document objects in the same order as the
        #                 BM25 index — needed to retrieve text by rank position.
        # reranker      : cross-encoder that re-scores (query, chunk) pairs
        #                 jointly for a more accurate final ranking.
        #                 Model is ~67 MB, downloads once, then cached locally.
        self.bm25_indexes = {}
        self.bm25_chunks  = {}
        self.reranker     = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # ─────────────────────────────────────────────────────────────────────

        self.TEST_UNIT_MAP = {
            "HEMOGLOBIN":   ["g/dl", "g/dl"],
            "ESR":          ["mm/hr"],
            "CREATININE":   ["mg/dl", "mmol/l"],
            "GLUCOSE":      ["mg/dl", "mmol/l"],
            "CHOLESTEROL":  ["mg/dl", "mmol/l"],
            "POTASSIUM":    ["mmol/l", "meq/l"],
            "SODIUM":       ["mmol/l", "meq/l"],
            "CALCIUM":      ["mg/dl", "mmol/l"],
            "VITAMIN":      ["ng/ml", "pg/ml", "nmol/l"],
            "IRON":         ["µg/dl", "ug/dl", "mcg/dl"],
            "PLATELET":     ["/cumm", "lakhs", "10^3"],
            "RBC":          ["mill/cumm", "10^6"],
            "WBC":          ["cells/ul", "/cumm", "10^3"],
        }


    def _load_and_classify_documents(self) -> list:
        """Load documents and classify by domain"""
        print("Loading and classifying documents...\n")

        loader = DirectoryLoader(
            path=self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()

        if not documents:
            print("No PDFs found in directory")
            return []

        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
            filename = os.path.basename(doc.metadata.get('source', 'unknown'))
            domain = self._classify_domain(doc.page_content, filename)
            doc.metadata.update({
                "domain": domain.value,
                "filename": filename,
                'source': doc.metadata.get('source', ''),
                'pages': doc.metadata.get('total_pages', 1),
                'section': "",
            })
            self.assign_section_metadata(doc)
            self.documents_by_domain[domain].append(doc)

        return documents


    def assign_section_metadata(self, doc):
        text = doc.page_content.lower()
        raw_text = doc.page_content

        has_numeric_results = bool(re.search(
            r'\b\d+\.?\d*\s*(mg/dl|g/dl|mmol|iu/l|u/l|%|fl|pg|mm/hr|cells/ul|meq/l|ng/ml|mcg)',
            raw_text, re.IGNORECASE
        )) or bool(re.search(
            r'\b(mm/hr|mg/dl|g/dl|iu/l|ng/ml|mcg|meq/l)\b',
            raw_text, re.IGNORECASE
        ))

        has_reference_range = bool(re.search(
            r'(\d+\.?\d*\s*[-–]\s*\d+\.?\d*|[<>]\s*\d+\.?\d*)',
            raw_text
        ))

        has_lab_headers = any(phrase in text for phrase in [
            "reference range", "normal range", "observed value",
            "test name", "result", "units", "methodology", "flag", "reference", "referral", "high", "low", "range"
        ])

        has_patient_block = any(phrase in text for phrase in [
            "patient name", "patient id", "date of birth",
            "sample collected", "referred by", "lab no", "report date"
        ])

        has_experience = any(phrase in text for phrase in [
            "professional experience", "professional summary", "previous experience", 
            "experience summary", "experience details", "experience overview",
            "past experience", "work experience", "employment history", "career history", 
            "experience highlights", "experience section", "experience breakdown"
        ])

        total_experience = re.search(r'(\d+(?:\.\d+)?)\s*\+?\s*years', text);

        has_date_range = bool(re.search(
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]+\d{4}'
        r'\s*(?:–|-|to)\s*'
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|current)',
        text, re.IGNORECASE
    ))
        has_resume_structure = any(phrase in text for phrase in [
            "tech stack", "tools & methodologies", "skills", "technologies"
        ])
        
        has_education = "education" in text
        has_contact = any(word in text for word in ["email", "phone", "linkedin", "github"])

        lab_signal_count = sum([has_numeric_results, has_reference_range, has_lab_headers])

        if lab_signal_count >= 2:
            doc.metadata["section"] = "lab_info"
        elif has_patient_block and lab_signal_count == 0:
            doc.metadata["section"] = "patient_information"
        elif has_experience and total_experience and not has_date_range:
            doc.metadata["section"] = "total_experience"
        elif has_date_range and has_experience:
            doc.metadata["section"] = "experience"
        elif has_resume_structure and not has_date_range:
            doc.metadata["section"] = "skills"
        elif has_education:
            doc.metadata["section"] = "education"
        elif has_contact:
            doc.metadata["section"] = "contact"
        else:
            doc.metadata["section"] = "general"

        return doc


    def _classify_domain(self, text: str, filename: str):
        if "resume" in filename.lower() or "cv" in filename.lower():
            return DomainNames.RESUME
        elif "medical" in filename.lower() or "report" in filename.lower() or "lab" in filename.lower():
            return DomainNames.MEDICAL
        else:
            return DomainNames.OTHER


    def detect_query_domain(self, query):
        query = query.lower()
        resume_words = ["skill", "experience", "education", "resume", "candidate", "tech stack"]
        medical_words = ["test", "hemoglobin", "lab", "report", "diagnosis", "patient", "treatment", "value", "result", "level"]

        if any(word in query for word in resume_words):
            return DomainNames.RESUME
        if any(word in query for word in medical_words):
            return DomainNames.MEDICAL
        return None


    def _save_documents(self):
        documents = self._load_and_classify_documents()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150
        )

        for domain, doc in self.documents_by_domain.items():
            if not doc:
                continue
            chunks = splitter.split_documents(doc)

            for chunk in chunks:
                chunk.metadata.setdefault("domain", domain.value)
                chunk.metadata.setdefault("section", "general")
                chunk.metadata.setdefault("filename", "unknown")

            db = FAISS.from_documents(chunks, self.embeddings)
            index_path = f"faiss_{domain.value}_index"
            db.save_local(index_path)
            self.vectorstores[domain] = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retrievers[domain] = self.vectorstores[domain].as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 20}
            )

            # ── HYBRID SEARCH: build BM25 index for this domain ──────────────
            # We use the same `chunks` list that just went into FAISS, so both
            # indexes always contain exactly the same documents.
            #
            # Tokenization: split each chunk's text into lowercase words.
            # BM25Okapi needs a list-of-lists:
            #   [["revenue", "grew", "20%"], ["patient", "name", "john"], ...]
            #
            # bm25_chunks[domain] stores the Document objects in the SAME ORDER
            # as the BM25 index so we can map rank → Document later.
            tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
            self.bm25_indexes[domain] = BM25Okapi(tokenized_corpus)
            self.bm25_chunks[domain]  = chunks
            # ─────────────────────────────────────────────────────────────────

        return self.retrievers


    def search_all_domains(self, query):
        """Fallback: search across all indexed domains."""
        all_docs = []
        for domain, retriever in self.retrievers.items():
            try:
                docs = retriever.invoke(query)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error searching domain {domain}: {e}")
        return all_docs


    # ── HYBRID SEARCH: BM25 search per domain ────────────────────────────────
    def _bm25_search(self, domain, query: str, top_k: int = 20):
        """
        Search the BM25 index for a given domain and return top_k Documents.

        How it works:
          1. Tokenize the query the same way the corpus was tokenized (lowercase split).
          2. BM25 scores every chunk in this domain's index.
          3. np.argsort gives positions sorted ascending; [::-1] reverses to
             descending (best first); [:top_k] keeps the top results.
          4. We only return chunks with a score > 0 (at least one keyword matched).
        """
        if domain not in self.bm25_indexes:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25_indexes[domain].get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            self.bm25_chunks[domain][i]
            for i in top_indices
            if scores[i] > 0   # skip chunks with zero keyword overlap
        ]


    # ── HYBRID SEARCH: Reciprocal Rank Fusion ────────────────────────────────
    def _rrf_merge(self, dense_docs, sparse_docs, k: int = 60, top_n: int = 20):
        """
        Merge two ranked lists (FAISS + BM25) into one using RRF.

        RRF ignores raw scores entirely and works only on rank (position).
        Formula:  score(doc) = Σ  1 / (k + rank_in_list)

        A document that appears near the top of BOTH lists accumulates the
        highest RRF score — meaning both retrievers agreed it is relevant.

        We use the first 200 characters of page_content as a unique key to
        detect when FAISS and BM25 returned the same chunk, even though they
        are different Python objects.
        """
        rrf_scores = {}
        doc_map    = {}

        for rank, doc in enumerate(dense_docs, start=1):
            key = doc.page_content[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

        for rank, doc in enumerate(sparse_docs, start=1):
            key = doc.page_content[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:top_n]]


    # ── HYBRID SEARCH: Cross-encoder re-ranking ──────────────────────────────
    def _rerank(self, query: str, docs, top_n: int = 5):
        """
        Re-score the merged candidates using a cross-encoder model.

        Unlike FAISS which encodes query and document SEPARATELY, a cross-encoder
        reads both together — making it far more accurate at judging relevance.
        We only run it on the small merged candidate set (not the full corpus)
        because it's slower than vector search.

        Returns the top_n most relevant documents according to the cross-encoder.
        """
        if not docs:
            return docs

        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        # zip pairs each doc with its score, sort descending, keep top_n
        reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in reranked[:top_n]]
    # ─────────────────────────────────────────────────────────────────────────


    def smart_search(self, query):
        domain  = self.detect_query_domain(query)
        section = self.detect_query_section(query)

        print(f"Detected domain: {domain}")
        print(f"Detected section: {section}")

      
        retriever_obj = self.retrievers[domain]
        if not hasattr(retriever_obj, 'invoke'):
            return self.search_all_domains(query)

        # Existing FAISS retrieval — unchanged
        dense_docs = retriever_obj.invoke(query)

        # ── HYBRID SEARCH: add BM25 + RRF ────────────────────────────────────
        sparse_docs = self._bm25_search(domain, query, top_k=20)
        docs        = self._rrf_merge(dense_docs, sparse_docs)
        # ─────────────────────────────────────────────────────────────────────

        # Existing section filtering — unchanged
        if section:
            filtered_docs = [d for d in docs if d.metadata.get("section") == section]
            if filtered_docs:
                docs = filtered_docs

        return docs


    # def extract_lab_value(self, query, docs):
    #     stop_words = {"what", "is", "the", "are", "value", "of", "my",
    #                   "result", "level", "show", "tell", "give", "me"}

    #     words = re.findall(r'\b[a-zA-Z]{2,}\b', query)
    #     test_name = None
    #     for word in words:
    #         if word.lower() not in stop_words:
    #             test_name = word.upper()
    #             break

    #     if not test_name:
    #         return None

    #     aliases = {
    #         "HEMOGLOBIN": ["HEMOGLOBIN", "HAEMOGLOBIN", "HGB", "HB"],
    #         "PLATELET":   ["PLATELET", "PLT"],
    #         "CREATININE": ["CREATININE", "CREAT", "CREATININE - SERUM"],
    #         "ESR":        ["ESR", "ERYTHROCYTE SEDIMENTATION RATE"],
    #     }
    #     search_names = aliases.get(test_name, [test_name])

    #     for doc in docs:
    #         lines = doc.page_content.split('\n')
    #         for line in lines:
    #             line_upper = line.upper()
    #             if not any(name in line_upper for name in search_names):
    #                 continue

    #             print(f"DEBUG matched line: {repr(line)}")

    #             numbers = re.findall(r'(?<![0-9\-\.])\b(\d+\.?\d*)\b(?!\s*[-–]\s*\d)', line)
    #             units = re.findall(
    #                 r'\b(g/dl|mg/dl|mm/hr|mmol/l|iu/l|u/l|%|fl|pg|ng/ml|µg/dl|µiu/ml|meq/l|gm/dl)\b',
    #                 line, re.IGNORECASE
    #             )

    #             range_match = re.search(r'^[\s<>]*([\d\.]+)\s*[-–]\s*([\d\.]+)', line)
    #             ref_range = f"{range_match.group(1)}-{range_match.group(2)}" if range_match else "not found"

    #             value = None
    #             for num in numbers:
    #                 if range_match and (num == range_match.group(1) or num == range_match.group(2)):
    #                     continue
    #                 value = num
    #                 break

    #             if not value:
    #                 continue

    #             unit = units[0] if units else ""
    #             result = f"{test_name}: {value} {unit}, Reference range: {ref_range}"
    #             print(f"DEBUG extracted: {result}")
    #             return result

    #     print(f"DEBUG: '{test_name}' not found in any chunk")
    #     return None


    def generate_answer(self, query, domainName, pdf_context, web_context=""):
        logger = Logging(query)
        try:
            with logger.setLatency("domain_detection"):
                domain = self.detect_query_domain(query)

            with logger.setLatency("section_detection"):
                section = self.detect_query_section(query)

            with logger.setLatency("retrieval"):
                docs = self.smart_search(query)

            if not docs:
                logger.setStatus("success")
                logger.log()
                return "No relevant documents found."

            # ── HYBRID SEARCH: re-rank the merged candidates ──────────────────
            # smart_search now returns up to 20 RRF-merged chunks.
            # _rerank cuts that to the 5 most relevant using the cross-encoder.
            with logger.setLatency("reranking"):
                docs = self._rerank(query, docs, top_n=5)
            # ─────────────────────────────────────────────────────────────────

            # for i, d in enumerate(docs):
            #     print(f"\n--- Chunk {i+1} ---")
            #     print(d.page_content)

            # context = "\n\n".join([
            #     f"[Chunk {i+1}]\n{d.page_content}"
            #     for i, d in enumerate(docs)
            # ])

            prompt = ""
            if domainName == DomainNames.MEDICAL.value:
                prompt = f"""You are a medical lab report assistant.

            IMPORTANT - THE DATA FORMAT IS REVERSED:
            Each row is formatted as: REFERENCE_RANGE  VALUE  TECHNOLOGY  TEST_NAME
            Example: "0.72-1.18mg/dL 0.86 PHOTOMETRYCREATININE - SERUM"
            - "0.72-1.18" is the REFERENCE RANGE (low-high)
            - "0.86" is the OBSERVED VALUE (the actual result)
            - "CREATININE - SERUM" is the TEST NAME

            STRICT RULES:
            1. Find the row where the test name "{query}" appears at the END of the line
            2. The OBSERVED VALUE is the STANDALONE number BEFORE the technology/method word
            3. The REFERENCE RANGE is the "X-Y" or "<X" pattern at the START of the line
            4. Never return the reference range as the value
            5. Return in this exact format: "Value: X unit (Reference range: LOW-HIGH)"
            6. If not found, say "Test not found"

            Context:
            {pdf_context}

            Question: {query}

            Answer:"""
            elif domainName == DomainNames.RESUME.value:
                parser = JsonOutputParser()
    
                format_instructions = """Return ONLY a valid JSON object in this exact format:
            {
                "answer": "direct answer to the question",
                "total_experience": "total years as number e.g. 12",
                "experience_breakdown": ["role 1 - duration", "role 2 - duration"]
            }
            Do not include any text outside the JSON object."""

                prompt = f"""You are a helpful assistant analysing a candidate's resume.
                
            STRICT RULES:
            - NEVER use "you", "your" — always say "The candidate" or "They"
            - Answer ONLY from the context provided
            - If not found say "I don't have enough information"

            {format_instructions}

            Context: {pdf_context}
            Question: {query}
            """
                
                
                # web_section = f"""
                # Additional information from the web:
                # {web_context}
                # """ if web_context else ""

                # instruction = (
                #     "Use BOTH the document and the web information to answer completely."
                #     if web_context else
                #     "Use the document context below to answer."
                # )

                # prompt = f"""You are a helpful assistant analysing a candidate's resume.
                # {instruction}

                # STRICT RULES — you MUST follow these without exception:
                # - NEVER use "you", "your", "you have", "you are" in your answer.
                # - ALWAYS replace "you" with "The candidate" or "They".
                # - CORRECT:   "The candidate has 2 years of AWS experience."
                # - INCORRECT: "You have 2 years of AWS experience."
                # - Do not mention chunks, documents, or your internal process.
                # - If the answer is not available, say "I don't have enough information."

                # Document context:
                # {pdf_context}
                # {web_section}
                # Question: {query}

                # Answer (remember: never use "you" or "your"):"""
            else:
                parser = StrOutputParser()   # ✅ define parser for else branch too
                prompt = f"""Answer the question based on the following context. 
            If the answer is not in the context, say "I don't know".

            Context: {pdf_context}
            Question: {query}
            """
            with logger.setLatency("llm_generation"):
                response = self.model.invoke(prompt)
                result = parser.parse(response.content) if domainName == DomainNames.RESUME.value else response.content
                logger.setStatus("success")
                logger.log()

            return result

        except Exception as e:
            logger.setStatus("error", str(e))
            logger.log()
            raise


    def detect_query_section(self, query):
        query_lower = query.lower()
        raw_query = query

        asking_for_lab_value = bool(re.search(
            r'\b(value|level|result|range|normal|high|low|count|rate|reading)\b',
            query_lower
        ))

        has_lab_abbreviation = bool(re.search(r'\b[A-Z]{2,}\b', raw_query))

        common_test_names = [
            "HEMOGLOBIN", "creatinine", "glucose", "cholesterol",
            "thyroid", "insulin", "calcium", "sodium", "potassium",
            "bilirubin", "albumin", "protein", "triglyceride", "urea"
        ]
        has_named_test = any(name in query_lower for name in common_test_names)

        if any(w in query_lower for w in ["skill", "tech stack", "technology", "tools"]):
            return "skills"
        if any(w in query_lower for w in ["past experience", "total experience"]):
            return "experience"
        if any(w in query_lower for w in ["experience", "worked", "job", "company"]):
            return "experience"
        if any(w in query_lower for w in ["education", "degree", "university", "college"]):
            return "education"
        if any(w in query_lower for w in ["contact", "email", "phone", "linkedin"]):
            return "contact"

        if (has_lab_abbreviation or has_named_test) and asking_for_lab_value:
            return "lab_info"
        if has_named_test or has_lab_abbreviation:
            return "lab_info"

        if any(w in query_lower for w in ["patient", "age", "sample", "collected"]):
            return "patient_information"

        return None


    def debug_pipeline(self, query):
        print("\n" + "="*60)
        print(f"QUERY: {query}")

        domain  = self.detect_query_domain(query)
        section = self.detect_query_section(query)
        print(f"Detected domain: {domain}")
        print(f"Detected section: {section}")

        if domain and domain in self.vectorstores:
            all_docs = self.vectorstores[domain].similarity_search(query, k=20)
            print(f"\nTotal chunks retrieved from FAISS: {len(all_docs)}")
            print("\nSection breakdown of retrieved chunks:")
            from collections import Counter
            section_counts = Counter(d.metadata.get("section") for d in all_docs)
            for sec, count in section_counts.items():
                print(f"  {sec}: {count} chunks")

            lab_chunks = [d for d in all_docs if d.metadata.get("section") == "lab_info"]
            print(f"\nlab_info chunks: {len(lab_chunks)}")
            for i, chunk in enumerate(lab_chunks):
                print(f"\n--- lab_info chunk {i+1} ---")
                print(chunk.page_content[:300])
        else:
            print(f"ERROR: domain '{domain}' not found in vectorstores")
            print(f"Available domains: {list(self.vectorstores.keys())}")

        print("="*60 + "\n")


    def clean_text(self, text):
        text = re.sub(r'(\d)([ ]*)([A-Z])', r'\1 \3', text)
        text = re.sub(r'([A-Z])(mg/dl|g/dl|mm/hr|mmol|iu/l|u/l|%|fl|pg|ng/ml)',
                      r'\1 \2', text, flags=re.IGNORECASE)
        text = re.sub(r'(mg\s*/\s*dl|mm\s*/\s*hr|g\s*/\s*dl)(\d)',
                      r'\1 \2', text, flags=re.IGNORECASE)
        text = re.sub(r'(mm|mg|g|iu|u|meq)\s*/\s*(dl|l|hr|ml)',
                      r'\1/\2', text, flags=re.IGNORECASE)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
    
