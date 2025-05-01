import os
import shutil
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def clear_folder(folder):
    """清空資料夾內所有檔案與子資料夾"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"清空 {file_path} 時發生錯誤：{e}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 向量庫儲存目錄設定
    vector_db_dir = os.path.join(current_dir, "../../knowledge/vector_db_TM")
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # 設定 base_folder、已處理資料夾與圖片快取資料夾
    base_folder = r"D:\MA system\LangGraph\knowledge\RAG data"
    processed_folder = os.path.join(base_folder, "已處理")
    image_cache_folder = os.path.join(base_folder, "cache")
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(image_cache_folder, exist_ok=True)
    
    # 初始化向量庫（可累積多個 PDF 的結果）
    embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
    vectorstore = Chroma(
        collection_name="mm_rag_clip_photos",
        embedding_function=embeddings,
        persist_directory=vector_db_dir
    )
    
    # 取得 base_folder 下所有 PDF 檔案（不包含子資料夾）
    pdf_files = [
        os.path.join(base_folder, fname)
        for fname in os.listdir(base_folder)
        if fname.lower().endswith(".pdf") and os.path.isfile(os.path.join(base_folder, fname))
    ]
    
    if not pdf_files:
        print("在 base_folder 中找不到 PDF 檔案。")
        return
    
    print("開始逐一處理 PDF 檔案...")
    for pdf_path in pdf_files:
        print(f"\n開始處理 PDF: {pdf_path}")
        
        # 清空圖片快取資料夾，避免前次的圖片影響當前解析
        clear_folder(image_cache_folder)
        
        # 解析 PDF，提取文字、圖片及表格元素，圖片儲存在 image_cache_folder
        try:
            raw_elements = partition_pdf(
                filename=pdf_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
                extract_image_block_output_dir=image_cache_folder,
            )
            print(f"解析完成 {pdf_path}，取得 {len(raw_elements)} 個元素。")
        except Exception as e:
            print(f"處理 {pdf_path} 時發生錯誤：{e}")
            continue  # 若解析失敗，略過此 PDF
        
        # --- 取得 image_cache_folder 中所有 .jpg 圖片的完整路徑 ---
        image_uris = sorted(
            [
                os.path.join(image_cache_folder, fname)
                for fname in os.listdir(image_cache_folder)
                if fname.lower().endswith(".jpg") and os.path.isfile(os.path.join(image_cache_folder, fname))
            ]
        )
        
        if image_uris:
            # 為每個圖片建立 metadata（僅存檔名）
            metadatas = [{"filename": os.path.basename(uri)} for uri in image_uris]
            print(f"開始 embed {len(image_uris)} 張圖片，來源目錄：{image_cache_folder} ...")
            vectorstore.add_images(uris=image_uris, metadatas=metadatas)
            
            # 儲存向量庫
            if hasattr(vectorstore, "persist"):
                vectorstore.persist()
            elif hasattr(vectorstore, "_client") and hasattr(vectorstore._client, "persist"):
                vectorstore._client.persist()
            print("圖片 embedding 完成。")
        else:
            print("在圖片快取資料夾中找不到 .jpg 圖片。")
        
        # 處理完畢後，將該 PDF 移動到 processed_folder
        try:
            dest_path = os.path.join(processed_folder, os.path.basename(pdf_path))
            shutil.move(pdf_path, dest_path)
            print(f"已將 {pdf_path} 移至 {dest_path}")
        except Exception as e:
            print(f"移動 {pdf_path} 時發生錯誤：{e}")
    
    print("\n所有 PDF 皆已處理完成。")

if __name__ == "__main__":
    main()