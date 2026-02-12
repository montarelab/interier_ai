from src.graph import build_graph


def test_build_graph_compiles_successfully():
    """Test that build_graph() compiles without errors."""
    graph = build_graph()

    assert graph is not None
    assert hasattr(graph, "nodes") or hasattr(graph, "get_graph")

    if hasattr(graph, "get_graph"):
        graph_structure = graph.get_graph()
        assert graph_structure is not None
