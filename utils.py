from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

# 可选：设置语言
wikipedia.set_lang("zh")


def get_clean_wikipedia_summary(subject, max_results=2):
    try:
        search_results = wikipedia.search(subject)
        filtered = []

        for title in search_results:
            # 只保留相关关键词的结果
            if any(keyword in title.lower() for keyword in ["模型", "openai", "ai", "视频", "text-to-video"]):
                try:
                    summary = wikipedia.summary(title, sentences=2)
                    filtered.append(f"【{title}】：{summary}")
                except (DisambiguationError, PageError):
                    continue

            if len(filtered) >= max_results:
                break

        return "\n\n".join(filtered) if filtered else "未找到可靠的维基百科信息。"

    except Exception as e:
        return f"Wikipedia 查询失败：{str(e)}"


def generate_script(subject, video_length, creativity, api_key):
    title_template = ChatPromptTemplate.from_messages(
        [
            ("human", "请为'{subject}'这个主题的视频想一个吸引人的标题")
        ]
    )

    script_template = ChatPromptTemplate.from_messages(
        [
            ("human",
             """你是一位短视频频道的博主。根据以下标题和相关信息，为短视频频道写一个视频脚本。
视频标题：{title}，视频时长：{duration}分钟，生成的脚本的长度尽量遵循视频时长的要求。
要求开头抓住眼球，中间提供干货内容，结尾有惊喜，脚本格式也请按照【开头、中间，结尾】分隔。
整体内容的表达方式要尽量轻松有趣，吸引年轻人。
脚本内容可以结合以下维基百科搜索出的信息，但仅作为参考，只结合相关的即可，对不相关的进行忽略：
```{wikipedia_search}```""")
        ]
    )

    model = ChatOpenAI(
        openai_api_key=api_key,
        base_url="https://api.aigc369.com/v1",  # 使用你提供的 API 代理地址
        temperature=creativity
    )

    # 生成标题
    title_chain = title_template | model
    title = title_chain.invoke({"subject": subject}).content

    # 获取干净的维基百科信息
    search_result = get_clean_wikipedia_summary(subject)

    # 生成脚本
    script_chain = script_template | model
    script = script_chain.invoke({
        "title": title,
        "duration": video_length,
        "wikipedia_search": search_result
    }).content

    return search_result, title, script


# 示例调用（你可以取消注释后直接测试）：
# import os
# result = generate_script("sora模型", 1, 0.7, os.getenv("OPENAI_API_KEY"))
# print("📘 参考信息：", result[0])
# print("\n🎬 视频标题：", result[1])
# print("\n📝 视频脚本：", result[2])