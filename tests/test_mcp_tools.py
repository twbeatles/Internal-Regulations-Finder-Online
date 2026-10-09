# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch

from app.mcp import tools as mcp_tools


def test_regulations_status_shape() -> None:
    with patch("app.mcp.tools.qa_system") as mock_qa:
        mock_qa.documents = ["a"]
        mock_qa.vector_store = None
        mock_qa.bm25 = MagicMock()
        out = mcp_tools.regulations_status()
    assert "ready" in out
    assert out["documents"] == 1


def test_regulations_search_delegates() -> None:
    with patch("app.mcp.tools.qa_system") as mock_qa:
        mock_qa.search.return_value = MagicMock(success=True, message="ok", data=[])
        out = mcp_tools.regulations_search("휴가", k=3)
    assert out["success"] is True
    mock_qa.search.assert_called_once()