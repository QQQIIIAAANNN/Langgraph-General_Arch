from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil

# 初始化 OpenAI 嵌入模型
load_dotenv() 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 初始化 Chroma 向量資料庫的存儲目錄
current_dir = os.path.dirname(os.path.abspath(__file__)) 
vector_db_dir = os.path.join(current_dir, "../../knowledge/vector_db_T") 
os.makedirs(vector_db_dir, exist_ok=True)

vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)

# 處理本地 PDF 文件
base_folder = os.path.join(current_dir, "../../knowledge/RAG data")  # 基本資料夾路徑
processed_folder = os.path.join(base_folder, "已處理")   # "已處理" 子資料夾路徑
os.makedirs(processed_folder, exist_ok=True)

# 查找 base_folder 中的所有 PDF 檔案（不包括子資料夾）
pdf_files = [f for f in os.listdir(base_folder) if f.endswith(".pdf") and os.path.isfile(os.path.join(base_folder, f))]

for pdf_file in pdf_files:
    pdf_path = os.path.join(base_folder, pdf_file)  # 完整 PDF 路徑
    try:
        # 加載 PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 拆分長文本為段落
        for doc in documents:
            doc.metadata["source"] = pdf_file  

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        # 將拆分的文檔嵌入並存入向量資料庫
        vectorstore.add_documents(split_docs)
        print(f"成功處理並嵌入文檔: {pdf_file}")

         # 移動 PDF 檔案到 "已處理" 子資料夾
        processed_path = os.path.join(processed_folder, pdf_file)
        shutil.move(pdf_path, processed_path)
        print(f"已將檔案移動到: {processed_path}")

    except Exception as e:
        print(f"處理 {pdf_file} 時發生錯誤: {e}")

# 儲存向量資料庫
vectorstore.persist()
print("向量資料庫建立完成，保存在: ", vector_db_dir)


# ###########################
# import os
# import uuid
# import base64
# import shutil
# from unstructured.partition.pdf import partition_pdf
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# from langchain.schema.messages import HumanMessage, SystemMessage
# from langchain.schema.document import Document
# from langchain_chroma import Chroma
# from dotenv import load_dotenv
# load_dotenv()

# # 配置路徑
# db_folder = "D:\\MA system\\LangGraph\\knowledge\\vector_db_T2"  # 向量資料庫的儲存位置
# base_folder = "D:\\MA system\\LangGraph\\knowledge\\RAG data"    # PDF 資料夾路徑 (資料夾中只包含 PDF 檔案)
# processed_folder = os.path.join(base_folder, "已處理")      # 已處理的 PDF 移動至此資料夾
# image_cache_folder = os.path.join(base_folder, "cache")        # PDF 擷取圖片或表格的暫存資料夾

# # 測試 PDF 配置
# fpath = base_folder  # 絕對路徑
# fname = "曲木構造案例集.pdf"
# pdf_path = os.path.join(fpath, fname)

# # 建立嵌入模型（使用 text-embedding-3-small）
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# # 初始化空的 Chroma 向量資料庫
# print("初始化 Chroma 向量資料庫...")
# vectorstore = Chroma(
#     embedding_function=embedding_model,
#     persist_directory=db_folder,
# )

# # 設定圖片輸出資料夾 (使用 image_cache_folder)
# output_path = image_cache_folder

# # 定義摘要函式：文字/表格摘要使用文字描述，圖片則傳入圖片 Base64 資料讓模型生成描述
# def summarize_text(text, element_type):
#     """
#     針對文字或表格元素生成摘要。
#     """
#     prompt_text = f"請針對以下{element_type}進行摘要:\n{text}\n" # Prompt 調整：更自然的中文指令
#     response = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024).invoke([HumanMessage(content=prompt_text)])
#     return response.content

# def encode_image(image_path):
#     """
#     將圖片編碼為 Base64 字串。
#     """
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def summarize_image(encoded_img):
#     """
#     針對圖片生成更精細的摘要，特別針對建築設計、構造等領域。
#     """
#     prompt = [
#         SystemMessage(content="""你是一位在建築設計和建築構造領域的專家。你的任務是分析建築文件中的圖片，並提供詳細且深入的摘要。"""), # 更專業的 SystemMessage
#         HumanMessage(content=[
#             {"type": "text", "text": """請詳細分析這張圖片，重點關注以下幾個方面：
#             1.  **建築設計理念和風格**: 描述圖片中展現的建築設計概念、風格流派、以及設計美學。
#             2.  **建築構造細節**:  分析圖片中的建築構造方式、結構系統、材料運用及其特性、以及節點細部處理。
#             3.  **細部大樣圖判讀**:  如果圖片是細部大樣圖，請解釋圖面所表達的構造細節、尺寸標註、材料圖例等資訊。
#             4.  **細節與裝飾**:  觀察建築的細部處理、裝飾元素、以及這些細節在整體設計中的作用。
#             5.  **學術性內容**:  如果圖片包含學術性圖表、分析圖、或是研究成果，請提取其核心觀點和資訊。
#             6.  **架構關係**:  分析圖片中各元素之間的空間關係、功能分區、流線組織，以及整體架構邏輯。

#             請針對以上重點，提供一份精確且資訊豐富的摘要。如果圖片包含多個元素或細節，請盡可能逐一描述。"""}, # 更精細的 HumanMessage
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
#         ])
#     ]
#     response = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024).invoke(prompt)
#     return response.content

# # 收集所有文件的 Document
# all_documents = []

# # 解析 PDF，提取文字、圖片及表格元素
# print("開始解析 PDF...")
# raw_pdf_elements = partition_pdf(
#     filename=pdf_path,
#     extract_images_in_pdf=True,
#     infer_table_structure=True,
#     chunking_strategy="by_title",
#     max_characters=4000,
#     new_after_n_chars=3800,
#     combine_text_under_n_chars=2000,
#     extract_image_block_output_dir=output_path,
# )
# print(f"PDF 解析完成，共取得 {len(raw_pdf_elements)} 個元素。")

# # 初始化各類元素的列表
# text_elements, table_elements = [], []
# image_paths = []  # 將從暫存資料夾中找到的圖片路徑收集起來
# text_summaries, table_summaries, image_summaries = [], [], []

# # 處理 PDF 元素 (文字及表格)
# for e in raw_pdf_elements:
#     rep = repr(e)
#     if "CompositeElement" in rep:
#         text = e.text
#         text_elements.append(text)
#         summary = summarize_text(text, "文字") # Element Type 中文化
#         text_summaries.append(summary)
#         print("→ 文字摘要生成完成。")
#     elif "Table" in rep:
#         table = e.text
#         table_elements.append(table)
#         summary = summarize_text(table, "表格") # Element Type 中文化
#         table_summaries.append(summary)
#         print("→ 表格摘要生成完成。")

# # 處理圖片：假設所有解析後的圖片都存放在 image_cache_folder
# # 如果有多個 PDF 同時使用相同的 cache 資料夾，可考慮用檔案修改時間或檔名來過濾
# for file in os.listdir(image_cache_folder):
#     if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#         img_path = os.path.join(image_cache_folder, file)
#         image_paths.append(img_path)
# # 對每張圖片進行摘要
# for img_path in image_paths:
#     encoded_img = encode_image(img_path)
#     image_summary = summarize_image(encoded_img)
#     image_summaries.append((encoded_img, image_summary))
#     print(f"→ 圖片摘要生成完成: {os.path.basename(img_path)}")

# # 建立 Document 物件
# documents = []
# for text, summary in zip(text_elements, text_summaries):
#     if summary.strip():
#         doc = Document(
#             page_content=summary,
#             metadata={
#                 "id": str(uuid.uuid4()),
#                 "type": "text",
#                 "original_content": text
#             }
#         )
#         documents.append(doc)
# for table, summary in zip(table_elements, table_summaries):
#     if summary.strip():
#         doc = Document(
#             page_content=summary,
#             metadata={
#                 "id": str(uuid.uuid4()),
#                 "type": "table",
#                 "original_content": table
#             }
#         )
#         documents.append(doc)
# for encoded_img, summary in image_summaries:
#     if summary.strip():
#         doc = Document(
#             page_content=summary,
#             metadata={
#                 "id": str(uuid.uuid4()),
#                 "type": "image",
#                 "original_content": encoded_img,  # 儲存 Base64 編碼資料
#                 "image_content": [ # 儲存圖片內容，以便前端顯示
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
#                 ]
#             }
#         )
#         documents.append(doc)

# print(f"從 {fname} 共建立 {len(documents)} 個文件。")
# all_documents.extend(documents)

# # PDF 處理完畢後，將原始 PDF 移至 processed_folder
# try:
#     shutil.move(pdf_path, os.path.join(processed_folder, fname))
#     print(f"已將 {fname} 移至已處理資料夾。")
# except Exception as e:
#     print(f"移動 {fname} 時發生錯誤: {e}")

# # 將所有文件加入向量資料庫
# if all_documents:
#     print("\n開始將文件加入向量資料庫...")
#     vectorstore.add_documents(all_documents)
#     print("文件成功加入向量資料庫。")
# else:
#     print("沒有文件可加入向量資料庫。")