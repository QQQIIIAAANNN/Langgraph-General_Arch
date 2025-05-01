from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

def get_long_term_store():
    """
    取得長期記憶存儲。
    這裡使用 InMemoryStore 作為範例，
    實際生產環境中可替換為 SQLite3、Redis 等持久化存儲。
    """
    return InMemoryStore()

def get_short_term_memory():
    """
    取得短期記憶檢查點管理器。
    LangGraph 內建短期記憶功能會透過檢查點自動管理狀態。
    """
    return MemorySaver()
