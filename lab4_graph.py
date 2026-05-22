from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lab4_config import LAB4_PLOTS_DIR, ensure_lab4_dirs
from lab4_linking import link_entity
from lab4_ner import Entity, run_ner


@dataclass
class GraphResult:
    path: str
    nodes: int
    edges: int

    def as_message(self) -> str:
        return f"Knowledge graph zapisany:\n{self.path}\nNodes: {self.nodes}\nEdges: {self.edges}"


def build_knowledge_graph(text: str, method: str = "spacy", language: str | None = None) -> GraphResult:
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("Brakuje networkx. Uruchom: python -m pip install -r requirements.txt") from exc

    ensure_lab4_dirs()
    ner_result = run_ner(method, text, language)
    graph = nx.Graph()

    enriched_entities = []
    for entity in ner_result.entities:
        candidates = link_entity(entity.text, ner_result.language, text)
        best = candidates[0] if candidates else None
        label = entity.text
        graph.add_node(
            label,
            kind=entity.label,
            wikidata=best.wikidata_id if best else "",
        )
        enriched_entities.append(entity)

    add_simple_edges(graph, enriched_entities)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LAB4_PLOTS_DIR / f"knowledge_graph_{timestamp}.png"
    draw_graph(graph, path)
    return GraphResult(str(path), graph.number_of_nodes(), graph.number_of_edges())


def add_simple_edges(graph, entities: list[Entity]) -> None:
    for left, right in zip(entities, entities[1:]):
        relation = relation_name(left, right)
        graph.add_edge(left.text, right.text, relation=relation)


def relation_name(left: Entity, right: Entity) -> str:
    if left.label == "PERSON" and right.label == "ORG":
        return "related_to_org"
    if left.label == "ORG" and right.label in {"GPE", "LOCATION"}:
        return "located_or_mentioned_in"
    if left.label == "PERSON" and right.label in {"GPE", "LOCATION"}:
        return "associated_with_place"
    return "co_occurs"


def draw_graph(graph, path) -> None:
    if graph.number_of_nodes() == 0:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "Brak encji do grafu", ha="center", va="center")
        plt.axis("off")
        plt.savefig(path, dpi=140)
        plt.close()
        return

    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("Brakuje networkx. Uruchom: python -m pip install -r requirements.txt") from exc

    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(graph, seed=42)
    labels = {
        node: f"{node}\n{data.get('kind', '')}"
        for node, data in graph.nodes(data=True)
    }
    edge_labels = nx.get_edge_attributes(graph, "relation")
    nx.draw_networkx_nodes(graph, pos, node_size=1800, node_color="#d7ecff")
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.75)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

