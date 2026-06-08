"""
agent.py
========
Wraps the existing MultiDomainRAG with agentic behaviour.

The agent adds two superpowers on top of plain RAG:

  1. CLARITY CHECK
     Before retrieving anything, the agent checks if the query is clear
     enough to search for. If not, it returns a follow-up question to the
     user instead of guessing.

     Example:
       User: "tell me about it"       → Agent: "What would you like to know?
                                                Your lab report or resume?"
       User: "what is my hemoglobin"  → Agent: proceeds to retrieval

  2. SUFFICIENCY CHECK + WEB SEARCH
     After retrieving chunks from the PDF, the agent checks if those chunks
     are actually enough to answer the question. If not, it automatically
     searches the web and combines the web result with the PDF context.

     Example:
       User: "is my cholesterol level dangerous?"
       PDF chunks have the value (4.8 mmol/L) but not medical interpretation.
       Agent: searches web for "is 4.8 mmol/L cholesterol dangerous"
              combines web explanation with PDF value → complete answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW THE LLM MAKES DECISIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We use STRUCTURED PROMPTS — prompts that instruct the LLM to reply with a
specific keyword (YES / NO / UNCLEAR) so our Python code can parse the
decision reliably.

  Bad prompt:  "Do you think the query is clear?"
  LLM might say: "Well, it depends on the context..."  ← unparseable

  Good prompt: "Reply with exactly one word: CLEAR or UNCLEAR."
  LLM says:    "UNCLEAR"  ← easy to parse with .strip().upper()

This is called CONSTRAINED GENERATION — guiding the LLM to output
something structured rather than free-form prose.

Install:
    pip install duckduckgo-search
"""

import logging
from dataclasses import dataclass

from ddgs import DDGS

from domains import DomainNames
from Logger import Logging

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AgentResponse — the result object returned to server.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The agent can return two types of response:
#
#   type = "clarification"
#     The query was unclear. The agent returns a follow-up question.
#     The frontend shows this question to the user.
#     answer           = the clarifying question text
#     used_web_search  = False (we didn't even get to retrieval)
#     pdf_was_enough   = None (not applicable)
#
#   type = "answer"
#     The agent has a real answer.
#     answer           = the LLM's response
#     used_web_search  = True if web search was needed
#     pdf_was_enough   = False if we had to go to the web
#
# @dataclass is a Python decorator that automatically generates __init__,
# __repr__ and other boilerplate for a class that just holds data.
# It saves us writing:  def __init__(self, type, answer, ...): self.type = type ...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AgentResponse:
    type: str                    # "clarification" or "answer"
    answer: str                  # answer text OR clarifying question text
    used_web_search: bool        # did we search the web?
    pdf_was_enough: bool | None  # None when type="clarification"
    chunks_used: int = 0  # optional: how many PDF chunks were used


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MultiDomainAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiDomainAgent:
    """
    An agent that wraps MultiDomainRAG and adds decision-making.

    It does NOT replace your existing RAG — it uses it.
    All your domain detection, section detection, hybrid search,
    and re-ranking still run exactly as before.

    Usage in server.py:
        from rag import MultiDomainRAG
        from agent import MultiDomainAgent

        rag   = MultiDomainRAG()
        agent = MultiDomainAgent(rag)

        # In the /api/query endpoint:
        result = agent.run(req.query, domain_str)
    """

    # Maximum number of web search results to fetch.
    # 3 is enough context without overwhelming the LLM's context window.
    WEB_SEARCH_MAX_RESULTS = 3

    def __init__(self, rag):
        """
        Args:
            rag: Your existing MultiDomainRAG instance.
                 We call its smart_search(), _rerank(), and model.invoke().
        """
        self.rag = rag


    # ── PUBLIC METHOD — called from server.py ────────────────────────────────

    def run(self, query: str, domain_str: str, history: list = []) -> AgentResponse:
        """
        Run the full agentic pipeline for one user query.

        Args:
            query:      The user's question.
            domain_str: Domain string from your existing detect_query_domain()
                        e.g. "MEDICAL", "RESUME", "GENERAL"

        Returns:
            AgentResponse with type="clarification" or type="answer".
        """
        log = Logging(query)

        try:

            with log.setLatency("history_processing"):
                 resolved_query = self._resolve_query(query, history)
                # Process conversation history if needed (not implemented in this example)
                
            # ── Step 1: Clarity check ─────────────────────────────────────────
            # Check if the query is specific enough to search for.
            # If not, return a clarifying question immediately.
            with log.setLatency("clarity_check"):
                clarifying_question = self._check_clarity(resolved_query)

            if clarifying_question:
                print("Query is unclear. Clarifying question:", clarifying_question)
                # Query is unclear — don't retrieve, just ask the user.
                logger.info("Query unclear. Returning clarifying question.")
                log.setStatus("success")
                log.log()
                return AgentResponse(
                    type="clarification",
                    answer=clarifying_question,
                    used_web_search=False,
                    pdf_was_enough=None,
                    chunks_used=None,
                )

            # ── Step 2: Retrieve from PDF (existing hybrid search + rerank) ───
            with log.setLatency("retrieval"):
                docs = self.rag.smart_search(resolved_query)

            if not docs:
                # PDF has no relevant content at all → go straight to web.
                logger.info("No PDF chunks found. Going to web search.")
                return self._answer_from_web_only(resolved_query, domain_str, log)

            with log.setLatency("reranking"):
                docs = self.rag._rerank(resolved_query, docs, top_n=5)

            # ── Step 3: Sufficiency check ─────────────────────────────────────
            # Ask the LLM: are these chunks enough to answer the question fully?
            with log.setLatency("sufficiency_check"):
                pdf_is_enough = self._check_sufficiency(resolved_query, docs)

            # ── Step 4a: Answer from PDF alone ────────────────────────────────
            if pdf_is_enough:
                logger.info("PDF context is sufficient. Answering from PDF.")
                print("logs", "PDF context is sufficient. Answering from PDF.")
                with log.setLatency("llm_generation"):
                    answer = self._generate_answer(
                        query=resolved_query,
                        domain_str=domain_str,
                        pdf_context=self._docs_to_context(docs),
                        web_context="",
                    )
                log.setStatus("success")
                log.log()
                return AgentResponse(
                    type="answer",
                    answer=answer,
                    used_web_search=False,
                    pdf_was_enough=True,
                    chunks_used=len(docs),
                )

            # ── Step 4b: PDF not enough → search web ─────────────────────────
            logger.info("PDF context insufficient. Searching the web.")
            print("logs", "PDF context insufficient. Searching the web.")
            with log.setLatency("web_search"):
                web_context = self._web_search(query)

            # ── Step 5: Answer from PDF + web combined ────────────────────────
            with log.setLatency("llm_generation"):
                answer = self._generate_answer(
                    query=resolved_query,
                    domain_str=domain_str,
                    pdf_context=self._docs_to_context(docs),
                    web_context=web_context,
                )

            log.setStatus("success")
            log.log()
            return AgentResponse(
                type="answer",
                answer=answer,
                used_web_search=True,
                pdf_was_enough=False,
            )

        except Exception as e:
            log.setStatus("error", str(e))
            log.log()
            raise


    def _resolve_query(self, query: str, history: list) -> str:

        # ── Short circuit: conversational messages ──
        conversational = [
            "bye", "goodbye", "thanks", "thank you", "ok", "okay",
            "cool", "great", "got it", "sure", "hello", "hi", "hey",
            "see you", "later", "stop", "exit", "quit"
        ]
        if query.strip().lower().rstrip("!.,") in conversational:
            return query

        if not history:
            return query

        # ── Check if query contains vague references ──
        vague_triggers = [
            "tell me more", "more about it", "more about that",
            "explain it", "explain that", "explain this",
            "elaborate", "go on", "continue", "and then",
            "what about it", "what about that", "tell me about it",
            "more details", "more info", "what else",
        ]

        query_lower = query.strip().lower()
        is_vague = any(trigger in query_lower for trigger in vague_triggers)

        # also catch very short queries with pronouns
        vague_pronouns = ["it", "that", "this", "them", "those", "these"]
        is_short_with_pronoun = (
            len(query.split()) <= 6 and
            any(f" {p} " in f" {query_lower} " for p in vague_pronouns)
        )

        if not (is_vague or is_short_with_pronoun):
            return query  # already specific enough, no need to resolve

        # ── Extract topic from last assistant message ──
        last_assistant = next(
            (m['content'] for m in reversed(history) if m['role'] == 'assistant'),
            None
        )
        last_user = next(
            (m['content'] for m in reversed(history) if m['role'] == 'user'),
            None
        )

        if not last_assistant:
            return query

        # take first sentence of last assistant reply as the topic
        topic = last_assistant.split('.')[0].strip()

        # cap topic length so the query doesn't become huge
        if len(topic) > 120:
            topic = topic[:120].rsplit(' ', 1)[0] + "..."

        # build the resolved query
        resolved = f"{query.rstrip('?.')} about: {topic}"
        print(f"DEBUG resolved: '{query}' → '{resolved}'")
        return resolved


    def _check_clarity(self, query: str):

        query = query.strip().lower()

        if not query or len(query) < 2:
            return "Please enter a question."

        vague_only = {
            "ok", "okay", "yes", "no",
            "help", "tell", "show", "hi", "hello", "hey",
             "thanks", "thank you", "cool", "great", "got it", "sure",
             "bye", "goodbye", "see you", "later", "stop", "exit", "quit"
        }

        words = set(query.split())

        if words.issubset(vague_only):
            return "Could you be more specific?"

        return None


    # ── STEP 3: Sufficiency check ─────────────────────────────────────────────
    #
    # WHY CHECK SUFFICIENCY?
    # Sometimes the PDF has partial information. For example:
    #   - Lab report has the cholesterol value but no medical interpretation
    #   - Resume has the job title but no details about what the role involves
    #
    # In these cases, the LLM can retrieve the value but can't answer questions
    # like "is this dangerous?" or "what does this role typically require?".
    # Web search fills that gap.
    #
    # HOW WE CHECK:
    # We show the LLM the retrieved chunks and ask it directly:
    # "Can you fully answer this question from these chunks alone?"
    # The LLM replies YES or NO.
    # ─────────────────────────────────────────────────────────────────────────

    def _check_sufficiency(self, query: str, docs) -> bool:
        """
        Ask the LLM if the retrieved chunks are enough to answer the question.

        Returns:
            True  → PDF context is sufficient, no web search needed
            False → PDF context is insufficient, trigger web search
        """
        context = self._docs_to_context(docs)

        prompt = f"""You are evaluating whether a set of document chunks contains 
enough information to fully answer a user's question.

Question: "{query}"

Document chunks:
{context}

Can the question be answered fully and accurately using ONLY the information 
in the chunks above?

Rules:
- Reply YES if the chunks contain the specific information needed to answer.
- Reply NO if the chunks are missing key information, are vague, or only 
  partially address the question.
- Reply with exactly one word: YES or NO.

Reply:"""

        response = self.rag.model.invoke(prompt)
        text = response.content.strip().upper()

        # We check startswith rather than exact match because the LLM might
        # sometimes output "YES." or "YES, the chunks..." — startswith handles that.
        is_sufficient = text.startswith("YES")
        logger.info("Sufficiency check result: %s", "SUFFICIENT" if is_sufficient else "INSUFFICIENT")
        return is_sufficient


    # ── STEP 4b: Web search ───────────────────────────────────────────────────
    #
    # WHY DUCKDUCKGO?
    # - Free, no API key needed
    # - Good quality results
    # - The duckduckgo-search Python library is simple to use
    #
    # We search for the user's exact query. DuckDuckGo returns snippets
    # (short summaries) from the top web pages. We join the top 3 snippets
    # into one context string and pass it to the LLM alongside the PDF chunks.
    # ─────────────────────────────────────────────────────────────────────────

    def _web_search(self, query: str) -> str:
        """
        Search the web using DuckDuckGo and return the top results as text.

        DDGS() is the DuckDuckGo search client.
        .text() performs a text search and returns a list of result dicts.
        Each dict has: "title", "href" (URL), "body" (snippet text).

        We only use "body" — the snippet text — as context for the LLM.
        """
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(query, max_results=self.WEB_SEARCH_MAX_RESULTS)
                )

            if not results:
                logger.warning("Web search returned no results for: %s", query)
                return ""

            # Join the snippets with a separator so the LLM can see
            # where one result ends and the next begins.
            web_text = "\n\n---\n\n".join(
                f"Source: {r.get('href', 'unknown')}\n{r.get('body', '')}"
                for r in results
            )
            logger.info("Web search returned %d results.", len(results))
            return web_text

        except Exception as e:
            # Web search failing should not crash the whole request.
            # We log the error and return empty string — the LLM will
            # answer from PDF context alone.
            logger.error("Web search failed: %s", e)
            return ""


    # ── STEP 5: Answer generation ─────────────────────────────────────────────
    #
    # This replaces the prompt logic that was in generate_answer() in rag.py.
    # The key difference from your existing prompts:
    #   - When web_context is empty  → answer only from PDF
    #   - When web_context has text  → tell the LLM it has both sources
    #     and ask it to combine them clearly
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_answer(
        self,
        query: str,
        domain_str: str,
        pdf_context: str,
        web_context: str,
    ) -> str:
        """
        Generate the final answer using PDF context, web context, or both.
        """
        has_web = bool(web_context.strip())

        if domain_str == DomainNames.MEDICAL.value:
            prompt = self._medical_prompt(query, pdf_context, web_context, has_web)
        else:
            prompt = self._general_prompt(query, pdf_context, web_context, has_web)

        response = self.rag.model.invoke(prompt)
        return response.content.strip()


    def _medical_prompt(self, query, pdf_context, web_context, has_web) -> str:
        web_section = f"""
Web search results (for medical interpretation):
{web_context}
""" if has_web else ""

        instruction = (
            "Use BOTH the lab report data AND the web search results to give "
            "a complete answer including the value and its medical significance."
            if has_web else
            "Use the lab report data below to answer."
        )

        return f"""You are a medical lab report assistant.

IMPORTANT - THE DATA FORMAT IS REVERSED:
Each row is formatted as: REFERENCE_RANGE  VALUE  TECHNOLOGY  TEST_NAME
Example: "0.72-1.18mg/dL 0.86 PHOTOMETRYCREATININE - SERUM"
- "0.72-1.18" is the REFERENCE RANGE
- "0.86" is the OBSERVED VALUE
- "CREATININE - SERUM" is the TEST NAME

{instruction}

Lab report chunks:
{pdf_context}
{web_section}
Question: {query}

Answer:"""


    def _general_prompt(self, query, pdf_context, web_context, has_web) -> str:
        web_section = f"""
Additional information from the web:
{web_context}
""" if has_web else ""

        instruction = (
            "Use BOTH the document and the web information to answer completely."
            if has_web else
            "Use the document context below to answer."
        )

        return f"""You are a helpful assistant. {instruction}
Do not mention chunks, documents, or your internal process.
If the answer is not available in any source, say "I don't have enough information."

Document context:
{pdf_context}
{web_section}
Question: {query}

Answer:"""


    # ── HELPER: fallback when PDF has zero relevant chunks ────────────────────

    def _answer_from_web_only(
        self, query: str, domain_str: str, log: Logging
    ) -> AgentResponse:
        """Called when smart_search returns nothing at all."""
        with log.setLatency("web_search"):
            web_context = self._web_search(query)

        if not web_context:
            log.setStatus("success")
            log.log()
            return AgentResponse(
                type="answer",
                answer="I could not find relevant information in your document or on the web.",
                used_web_search=True,
                pdf_was_enough=False,
            )

        with log.setLatency("llm_generation"):
            answer = self._generate_answer(query, domain_str, "", web_context)

        log.setStatus("success")
        log.log()
        return AgentResponse(
            type="answer",
            answer=answer,
            used_web_search=True,
            pdf_was_enough=False,
        )


    # ── HELPER: convert Document list to context string ───────────────────────

    def _docs_to_context(self, docs) -> str:
        """
        Join retrieved Document objects into one context string for the LLM.
        Same format as your existing generate_answer() in rag.py.
        """
        return "\n\n".join([
            f"[Chunk {i+1}]\n{d.page_content}"
            for i, d in enumerate(docs)
        ])
