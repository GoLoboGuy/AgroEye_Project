# Backend/services/rag_service.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from Backend.config import settings

try:
    from openai import OpenAI  # optional
except Exception:
    OpenAI = None


@dataclass
class Retrieved:
    text: str
    meta: Dict
    score: float


class RagService:
    def __init__(self):
        self.index_dir = settings.RAG_INDEX_DIR
        self.emb = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        self.vs: FAISS | None = None
        self._load_index()

    def _load_index(self):
        if os.path.isdir(self.index_dir) and os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            self.vs = FAISS.load_local(self.index_dir, self.emb, allow_dangerous_deserialization=True)
        else:
            self.vs = None

    def search(self, query: str, k: int = 4) -> List[Retrieved]:
        if not self.vs:
            return []
        docs_scores = self.vs.similarity_search_with_score(query, k=k)
        out: List[Retrieved] = []
        for doc, score in docs_scores:
            out.append(Retrieved(text=doc.page_content, meta=doc.metadata or {}, score=float(score)))
        return out

    def make_sources(self, items: List[Retrieved]) -> List[str]:
        sources = []
        for r in items:
            src = r.meta.get("source")
            page = r.meta.get("page")
            sheet = r.meta.get("sheet")
            if src and page is not None:
                sources.append(f"{src}#p{page}")
            elif src and sheet:
                sources.append(f"{src}#sheet:{sheet}")
            elif src:
                sources.append(str(src))
        # uniq
        uniq, seen = [], set()
        for s in sources:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        return uniq

    def generate_explanation(self, query: str, items: List[Retrieved]) -> str:
        """OPENAI_API_KEY 있으면 LLM, 없으면 간이 요약.
        반환 문자열 끝에 \n\n출처: - file#pN ... 형태 첨부.
        """
        sources = self.make_sources(items)
        context = "\n\n".join([r.text for r in items])[:6000]

        if settings.OPENAI_API_KEY and OpenAI is not None:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            sys = (
                "너는 농작물 병해충 설명 도우미다. 받은 컨텍스트만 바탕으로 정확하고 간결한 한국어 설명을 작성해라.\n"
                "농가 실무자가 이해하기 쉽게 원인, 증상, 현장 확인 팁, 방제/관리 요점을 bullet로 정리하고, 추정/불확실 부분은 명시해라."
            )
            user = (
                f"질의: {query}\n\n"
                f"컨텍스트:\n{context}\n\n"
                "요구사항:\n- 핵심 요약(3~5줄)\n- 증상 체크리스트\n- 방제/관리 요점(실행 가능한 문장)\n- 필요한 경우 참고 기준/수치 포함\n"
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                    temperature=0.2,
                )
                body = resp.choices[0].message.content.strip()
            except Exception as e:
                body = f"[LLM 호출 실패: {e}]\n\n" + (context[:1200] or "")
        else:
            # 간이: 컨텍스트 앞부분을 요약처럼 제공
            head = context[:1000]
            body = (
                "[LLM 미사용: 간이 안내]\n"
                "아래 컨텍스트를 기반으로 추정한 요점입니다. LLM 키 설정 시 품질이 개선됩니다.\n\n"
                f"{head}"
            )

        if sources:
            body += "\n\n출처:\n" + "\n".join([f"- {s}" for s in sources])
        return body

rag = RagService()