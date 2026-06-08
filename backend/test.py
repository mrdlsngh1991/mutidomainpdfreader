
from asyncio.log import logger

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from domains import DomainNames
import os
from collections import defaultdict
from llm import load_model
from Logger import Logging
import re

#from main import query

class MultiDomainRAG:

    def __init__(self):
        self.pdf_directory = "pdfs/"
        self.documents_by_domain = defaultdict(list)
        self.document_metadata = defaultdict(list)
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2');
        self.retrievers = {}
        self.vectorstores = {}
        tokenizer, model =  load_model()
        self.model = model;
        self.bm5Retrievers = {}
        self.prompt_domain = '';
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
            path= self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        if not documents:
            print("No PDFs found in directory")
            return []
        
       # print(f"Loaded {len(documents)} documents\n")
        
        # Classify each document
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
            # Get filename for tracking
            filename = os.path.basename(doc.metadata.get('source', 'unknown'))
           # print("file name " , filename);
            
            # Classify domain
            domain = self._classify_domain(doc.page_content, filename)
            doc.metadata.update({
                "domain": domain.value,
                "filename": filename,
                'source': doc.metadata.get('source', ''),
                'pages': doc.metadata.get('total_pages', 1),
                'section': "",
            })
            # print("domain value", domain)
            # Store metadata
            
            self.assign_section_metadata(doc)
           # if doc.metadata["section"] == "contact":
            
            #    print("doc details--->",doc);
            # Track by domain
            self.documents_by_domain[domain].append(doc)
           
            
           # print("here is the doc metadata", doc)
            # print(f"     → Pages: {doc.metadata.get('source', 1)}\n")
            
        return documents


    # def assign_section_metadata(self, doc):

    #     text = doc.page_content.lower()
    
    #     if "tech stack" in text or "tools & methodologies" in text or "skills" in text:
    #         doc.metadata["section"] = "skills"
    
    #     elif "professional experience" in text:
    #         doc.metadata["section"] = "experience"
    
    #     elif "education" in text:
    #         doc.metadata["section"] = "education"

    #     elif "email" in text or "contact" in text or "phone" in text:
    #         doc.metadata["section"] = "contact"

    #     elif "summary" in text:
    #         doc.metadata["section"] = "summary"
        
    #     elif "patient name" in text or "Age" in text:
    #         doc.metadata["section"] = "patient_information"

    #     elif any(word in text for word in [
    #         "test name",
    #         "reference range",
    #         "result",
    #         "hemoglobin",
    #         "rbc",
    #         "wbc",
    #         "platelet",
    #         "vitamin",
    #         "calcium",
    #         "observed value"
    #     ]):
    #         doc.metadata["section"] = "lab_info"
    #     else:
    #         doc.metadata["section"] = "general"
    
    #     return doc


    

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
            "test name", "result", "units", "methodology", "flag", "reference" , "referral", "high" , "low", "range"
        ])
    
        has_patient_block = any(phrase in text for phrase in [
            "patient name", "patient id", "date of birth",
            "sample collected", "referred by", "lab no", "report date"
        ])
    
        has_resume_structure = any(phrase in text for phrase in [
            "tech stack", "tools & methodologies", "skills", "technologies"
        ])
        has_experience = "professional experience" in text
        has_education = "education" in text
        has_contact = any(word in text for word in ["email", "phone", "linkedin", "github"])
    
        # Count how many lab signals are present
        lab_signal_count = sum([has_numeric_results, has_reference_range, has_lab_headers])
    
        # Lab structure check comes BEFORE patient block check
        if lab_signal_count >= 2:
            doc.metadata["section"] = "lab_info"
    
        # Patient block only wins if there are NO lab signals
        elif has_patient_block and lab_signal_count == 0:
            doc.metadata["section"] = "patient_information"
    
        elif has_resume_structure:
            doc.metadata["section"] = "skills"
    
        elif has_experience:
            doc.metadata["section"] = "experience"
    
        elif has_education:
            doc.metadata["section"] = "education"
    
        elif has_contact:
            doc.metadata["section"] = "contact"
    
        else:
            doc.metadata["section"] = "general"
    
        return doc

    def _classify_domain (self, text: str , filename: str) -> () :
        if "resume" in filename.lower():
            return DomainNames.RESUME
        elif "medical" in filename.lower() or "report" in filename.lower() or "lab" in filename.lower():
            return DomainNames.MEDICAL
        else:
            return DomainNames.OTHER


    def detect_query_domain(self, query):

        query = query.lower()

        resume_words = ["skill","experience","education","resume","candidate","tech stack"]
        medical_words = ["test","hemoglobin","lab","report","diagnosis","patient","treatment", "value", "result" , "level"]

        if any(word in query for word in resume_words):
            return DomainNames.RESUME
    
        if any(word in query for word in medical_words):
            return DomainNames.MEDICAL
    
        return None
    
    def _save_documents (self) :
        documents = self._load_and_classify_documents();
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        for domain, doc in self.documents_by_domain.items():
            if not doc:
                continue; 
            chunks = splitter.split_documents(doc)

            for chunk in chunks:
                chunk.metadata.setdefault("domain", domain.value)
                chunk.metadata.setdefault("section", "general")
                chunk.metadata.setdefault("filename", "unknown")
            
            db = FAISS.from_documents(chunks, self.embeddings);
            index_path = f"faiss_{domain.value}_index"
            db.save_local(index_path)
            self.vectorstores[domain] = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
            # Connect query to FAISS index using a retriever
            self.retrievers[domain] = self.vectorstores[domain].as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "fetch_k": 20
                }
            )
        return self.retrievers;


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

    def smart_search(self, query):
      
        domain = self.detect_query_domain(query)
        
      
        section = self.detect_query_section(query)
        
        print(f"Detected domain: {domain}")
        print(f"Detected section: {section}")
        if not domain or domain not in self.retrievers:
            return self.search_all_domains(query)
    
        retriever_obj = self.retrievers[domain]

        if not hasattr(retriever_obj, 'invoke'):
            return self.search_all_domains(query)
        

        docs = retriever_obj.invoke(query)
    
        if section:
            filtered_docs = [d for d in docs if d.metadata.get("section") == section]
            if filtered_docs:          # only override if we found matches
                docs = filtered_docs
    
        return docs

    


    # Define expected units for known tests
    def extract_lab_value(self, query, docs):
        stop_words = {"what", "is", "the", "are", "value", "of", "my",
                    "result", "level", "show", "tell", "give", "me"}

        words = re.findall(r'\b[a-zA-Z]{2,}\b', query)
        test_name = None
        for word in words:
            if word.lower() not in stop_words:
                test_name = word.upper()
                break

        if not test_name:
            return None

        aliases = {
            "HEMOGLOBIN": ["HEMOGLOBIN", "HAEMOGLOBIN", "HGB", "HB"],
            "PLATELET":   ["PLATELET", "PLT"],
            "CREATININE": ["CREATININE", "CREAT", "CREATININE - SERUM"],
            "ESR":        ["ESR", "ERYTHROCYTE SEDIMENTATION RATE"],
        }
        search_names = aliases.get(test_name, [test_name])

        for doc in docs:
            lines = doc.page_content.split('\n')
            for line in lines:
                line_upper = line.upper()
                if not any(name in line_upper for name in search_names):
                    continue

                print(f"DEBUG matched line: {repr(line)}")

                # Format: RANGE  VALUE  TECHNOLOGY  TEST_NAME
                # Pull ALL standalone numbers from the line
                numbers = re.findall(r'(?<![0-9\-\.])\b(\d+\.?\d*)\b(?!\s*[-–]\s*\d)', line)

                units = re.findall(
                    r'\b(g/dl|mg/dl|mm/hr|mmol/l|iu/l|u/l|%|fl|pg|ng/ml|µg/dl|µiu/ml|meq/l|gm/dl)\b',
                    line, re.IGNORECASE
                )

                # Reference range is X-Y at the start
                range_match = re.search(r'^[\s<>]*([\d\.]+)\s*[-–]\s*([\d\.]+)', line)
                ref_range = f"{range_match.group(1)}-{range_match.group(2)}" if range_match else "not found"

                # The observed value is a standalone number NOT part of the range
                # It appears AFTER the range and BEFORE the technology word
                value = None
                for num in numbers:
                    # Skip numbers that are part of the reference range
                    if range_match and (num == range_match.group(1) or num == range_match.group(2)):
                        continue
                    value = num
                    break

                if not value:
                    continue

                unit = units[0] if units else ""
                result = f"{test_name}: {value} {unit}, Reference range: {ref_range}"
                print(f"DEBUG extracted: {result}")
                return result

        print(f"DEBUG: '{test_name}' not found in any chunk")
        return None

    def generate_answer(self, query, domainName):
        logger = Logging(query);
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
        
            for i, d in enumerate(docs):
                print(f"\n--- Chunk {i+1} ---")
                print(d.page_content)

            context = "\n\n".join([
                f"[Chunk {i+1}]\n{d.page_content}"
                for i, d in enumerate(docs)
            ])
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
            {context}

            Question: {query}

            Answer:"""
            else:
                prompt = f"""
                You are a helpful assistant. Use the context below to answer the question clearly and concisely.
                Do not mention chunks, documents, or your internal process.
                If the answer is not in the context, say "I don't have enough information to answer that."

                Context:
                {context}

                Question:
                {query}

                Answer:
                """
            with logger.setLatency("llm_generation"):
                response = self.model.invoke(prompt)
                logger.setStatus("success")
                logger.log()
            return response
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
    
        # Match both "ESR" (uppercase) and "hemoglobin" (lowercase medical word)
        has_lab_abbreviation = bool(re.search(r'\b[A-Z]{2,}\b', raw_query))
        
        # Also check lowercase medical test names
        common_test_names = [
            "HEMOGLOBIN", "creatinine", "glucose", "cholesterol",
            "thyroid", "insulin", "calcium", "sodium", "potassium",
            "bilirubin", "albumin", "protein", "triglyceride", "urea"
        ]
        has_named_test = any(name in query_lower for name in common_test_names)
    
        if any(w in query_lower for w in ["skill", "tech stack", "technology", "tools"]):
            return "skills"
        if any(w in query_lower for w in ["experience", "worked", "job", "company"]):
            return "experience"
        if any(w in query_lower for w in ["education", "degree", "university", "college"]):
            return "education"
        if any(w in query_lower for w in ["contact", "email", "phone", "linkedin"]):
            return "contact"
    
        # Catches "ESR", "TSH", "HBA1C" AND "hemoglobin", "creatinine"
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
        
        # Step 1: Check domain detection
        domain = self.detect_query_domain(query)
        section = self.detect_query_section(query)
        print(f"Detected domain: {domain}")
        print(f"Detected section: {section}")
        
        # Step 2: Check what chunks exist for this domain
        if domain and domain in self.vectorstores:
            all_docs = self.vectorstores[domain].similarity_search(query, k=20)
            print(f"\nTotal chunks retrieved from FAISS: {len(all_docs)}")
            print("\nSection breakdown of retrieved chunks:")
            from collections import Counter
            section_counts = Counter(d.metadata.get("section") for d in all_docs)
            for sec, count in section_counts.items():
                print(f"  {sec}: {count} chunks")
            
            # Step 3: Show what lab_info chunks actually contain
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
        # Fix "30ERYTHROCYTE" → "30 ERYTHROCYTE"
        text = re.sub(r'(\d)([ ]*)([A-Z])', r'\1 \3', text)
        
        # Fix "RATEmm/hr" → "RATE mm/hr"  
        text = re.sub(r'([A-Z])(mg/dl|g/dl|mm/hr|mmol|iu/l|u/l|%|fl|pg|ng/ml)', 
                      r'\1 \2', text, flags=re.IGNORECASE)
        
        # Fix "mm / hr30" → "mm/hr 30"
        text = re.sub(r'(mg\s*/\s*dl|mm\s*/\s*hr|g\s*/\s*dl)(\d)', 
                      r'\1 \2', text, flags=re.IGNORECASE)
        
        # Remove extra spaces within units: "mm / hr" → "mm/hr"
        text = re.sub(r'(mm|mg|g|iu|u|meq)\s*/\s*(dl|l|hr|ml)', 
                      r'\1/\2', text, flags=re.IGNORECASE)
        
        # Collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

# rag = MultiDomainRAG();
# retrievers = rag._save_documents();
# #rag.debug_pipeline("give me values of all the tests which are not in the reference range?")
# docs = rag.generate_answer("give me values of all the tests which are not in the reference range");

# print("response is --->", docs);

