import os
import json
import re
from typing import List, Dict, Any
import networkx as nx
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
from graph.schema import (
    EntityType,
    RelationType,
    ExtractedClinicalTriples,
    ClinicalEntity,
    ClinicalRelationship
)

load_dotenv()

GRAPH_SAVE_PATH = os.path.join("data", "processed", "clinical_graph.json")

EXTRACTION_SYSTEM_PROMPT = """
You are a Clinical Knowledge Graph Extraction Engine.
Extract factual clinical entities and relationships strictly from the provided FDA drug labeling text.

STRICT ONTOLOGY RULES:
1. Entity Types MUST be one of: "Drug", "Condition", "SideEffect".
2. Relation Types MUST be one of:
   - "TREATS": (Drug) -> TREATS -> (Condition)
   - "INTERACTS_WITH": (Drug) -> INTERACTS_WITH -> (Drug)
   - "CAUSES": (Drug) -> CAUSES -> (SideEffect)
   - "CONTRAINDICATIONS_IN": (Drug) -> CONTRAINDICATIONS_IN -> (Condition)
3. Normalize drug names to simple common names (e.g. use "Warfarin" instead of "Warfarin Sodium Tablets USP").
4. ONLY extract relationships directly supported by the text.

Output MUST be valid JSON adhering to the ExtractedClinicalTriples schema:
{
  "entities": [{"name": "...", "entity_type": "Drug|Condition|SideEffect"}],
  "relationships": [{"source": "...", "relation": "TREATS|INTERACTS_WITH|CAUSES|CONTRAINDICATIONS_IN", "target": "..."}]
}
"""


def normalize_entity_name(name: str) -> str:
    """
    Normalizes drug and clinical entity names (e.g., strips dosage, tablets, uppercase).
    """
    clean = name.strip().title()
    clean = re.sub(r'\b(Tablets|Capsules|Usp|Oral|Solution|Sodium)\b', '', clean, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', clean).strip()


class ClinicalGraphBuilder:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.graph = nx.DiGraph()

    def extract_triples_from_chunk(self, chunk_text: str, drug: str, section: str) -> ExtractedClinicalTriples:
        """
        Sends chunk text to Gemini and parses structured entities and relationships.
        """
        prompt = f"""
FDA LABEL SECTION: {section}
PRIMARY DRUG: {drug}

CLINICAL TEXT:
{chunk_text}

Extract entities and relationships in valid JSON:
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=ExtractedClinicalTriples,
                    temperature=0.0
                )
            )

            # Parse and validate with Pydantic
            data = json.loads(response.text)
            return ExtractedClinicalTriples(**data)
        except Exception as e:
            print(f"     [!] Extraction error on chunk: {e}")
            return ExtractedClinicalTriples()

    def build_graph_from_chunks(
        self,
        chunks_file: str = os.path.join("data", "processed", "chunks.json"),
        max_chunks_per_section: int = 3
    ) -> nx.DiGraph:
        """
        Processes key clinical sections and builds the NetworkX graph.
        """
        with open(chunks_file, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)

        # Target high-signal clinical sections
        target_sections = {
            "INDICATIONS AND USAGE",
            "CONTRAINDICATIONS",
            "WARNINGS AND PRECAUTIONS",
            "DRUG INTERACTIONS",
            "BOXED WARNING"
        }

        # Filter chunks by target sections, grouping by (drug, section)
        selected_chunks = []
        section_counts = {}

        for chunk in all_chunks:
            sec = chunk["metadata"]["section"]
            drug = chunk["metadata"]["drug"]
            key = (drug, sec)
            if sec in target_sections:
                count = section_counts.get(key, 0)
                if count < max_chunks_per_section:
                    selected_chunks.append(chunk)
                    section_counts[key] = count + 1

        print(f"[*] Extracting clinical triples from {len(selected_chunks)} targeted chunks...")

        for idx, chunk in enumerate(selected_chunks, start=1):
            drug = chunk["metadata"]["drug"]
            section = chunk["metadata"]["section"]
            text = chunk["text"]
            print(f"---> [{idx}/{len(selected_chunks)}] Extracting: {drug} | {section} (Page {chunk['metadata']['page']})")

            triples = self.extract_triples_from_chunk(text, drug, section)

            # Add entities as nodes
            for ent in triples.entities:
                normalized = normalize_entity_name(ent.name)
                if normalized:
                    self.graph.add_node(normalized, entity_type=ent.entity_type.value)

            # Add relationships as directed edges
            for rel in triples.relationships:
                src = normalize_entity_name(rel.source)
                tgt = normalize_entity_name(rel.target)
                if src and tgt and src != tgt:
                    self.graph.add_edge(src, tgt, relation=rel.relation.value)

            time.sleep(4.5)
                
        # Save to disk
        self.save_graph()
        return self.graph

    def save_graph(self, path: str = GRAPH_SAVE_PATH):
        """
        Serializes the NetworkX graph nodes and edges to JSON.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[✓] Knowledge Graph saved to: {path}")
        print(f"    Total Nodes (Entities):     {self.graph.number_of_nodes()}")
        print(f"    Total Edges (Relationships): {self.graph.number_of_edges()}")


if __name__ == "__main__":
    builder = ClinicalGraphBuilder()
    graph = builder.build_graph_from_chunks(max_chunks_per_section=2)