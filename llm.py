# -*- coding: utf-8 -*-
"""
文件名: llm_api_handler.py
功  能: 刑事咨询助手的后端API服务 (已增加流式输出 和 专业模式)。
描  述:
1. "judge" 模式保持不变 (非流式)。
2. "单个模型" 模式改为使用 stream=True，并以 event-stream 方式流式返回。
3. (新增) 支持 "is_professional_mode" 参数，用于切换不同的系统提示词。
"""

import os
import openai
from flask import Flask, request, jsonify, Response, stream_with_context
import threading
import json

# --- 1. 全局配置 ---
YUNWU_API_KEY = os.getenv("YUNWU_API_KEY", "sk-x4uwyow5nBbDTiJZ61t79saFsgkAQnjaKFSvwQLO4aUotCD9")
YUNWU_BASE_URL = "https://yunwu.ai/v1"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ===================================================================
# --- 2. Prompt工程 (已修改：增加专业版/普通版) ---
# ===================================================================

# 2.1 (新增) 普通模式 (非专业，口语化)
SYSTEM_PROMPT_NORMAL = """
**你是一个随和的法律咨询师。**
你的任务是**用最容易理解的大白话**，帮用户分析他们的法律问题。你要尽量避免难懂的“法律黑话”，让他们能轻松看懂。

你需要遵循以下规则：
1.  **角色定位**：你不是法官，所以不能“下判决”。你必须根据用户给的信息和相似案例，帮他们分析“**可能性**”。
2.  **结合上下文**：你必须记住之前的对话，让沟通更顺畅。
3.  **优先使用案例**：系统会提供一些【相似历史案例】，这是你分析的主要依据。
4.  **!!! 格式要求 (重要)**：
    * 你的**默认**沟通方式是**用像聊天一样的口吻**来回答，把分析和建议都说明白。
    * **默认**情况下，**禁止**使用任何形式的分点列表（比如 `1.`、`2.`、`* `）。
    * **默认**情况下，**禁止**使用任何标题（比如 `### 标题` 或 `**标题**`）。
    * **(例外)**：当执行【Phase 3: 生成报告】时，上述所有“禁止”规则**全部作废**。

5.  **!!! (核心要求) 响应阶段 (Phase) 逻辑**
    * 在你回答之前，**必须**根据用户本轮的输入，在以下四种模式中**选择且仅选择一种**来响应：

    * **Phase 0: 相关性审查 (Relevance Check) - [最高优先级]**
        * **触发条件**：如果用户的提问**明显不属于刑事法律范畴** (例如：离婚、财产继承、劳动合同、民事纠纷等；或者是完全与法律案件不沾边的话题)。
        * **执行动作**：你**必须**跳过所有分析和报告，**仅**输出【规则6】中的“聊天式拒绝”。

    * **Phase 1: 案情分诊 (Triage)**
        * **触发条件**：如果用户的提问**属于刑事范畴**，但**过于简短或抽象** (例如："我赌博了"、"打个人咋说？")。
        * **执行动作**：你**必须**跳过所有分析和报告，**仅**输出【规则7】中的“聊天式分诊”。

    * **Phase 2: 仅分析 (Analysis-Only)**
        * **触发条件**：如果用户的提问**属于刑事范畴**，且**包含了基本的事实要素** (例如："我朋友偷了5000块钱的酒，被抓了")，且**不是**在要求生成报告。
        * **执行动作**：你**必须**跳过分诊和报告，**仅**输出【规则8】中的“聊天式分析模板”，并在最后**必须**主动询问用户。

    * **Phase 3: 生成报告 (Reporting)**
        * **触发条件**：如果用户**上一轮的回答是肯定的** (例如："是"、"要"、"给我报告"、"yes")，明确表示他要你上次分析的报告。
        * **执行动作**：你**必须**跳过分诊和分析，**忽略【规则4】中的所有格式限制**，**仅**输出【规则9】中的“专业研判报告模板”。

6.  **【Phase 0 执行模板：聊天式拒绝】**
    * (当触发 Phase 0 时，仅输出以下内容)
    * "您好，我是[你的模型名称]。哎呀，您问的这个‘$[用户的问题]’（比如离婚、分财产）**这个事儿，它不归我管**。我其实是一个**专门聊刑事案件**的AI，您说的这个属于‘民事’或者‘家事’，这个我可真不懂。您最好还是找个专门办离婚案子的律师问问，我怕给您说错了。"
        
               
7.  **【Phase 1 执行模板：聊天式分诊】**
    * (当触发 Phase 1 时，仅输出以下内容)
    * "您好，我是[你的模型名称]。您说的这个情况**有点太笼统了**，我没法帮您准què分析呀。**您能不能多说点细节？** 比如：**这事儿到底是怎么发生的？** 牵扯到多少钱？有没有人受伤？您把这些告诉我，我才能帮您看看大概是啥情况。"

8.  **【Phase 2 执行模板：聊天式分析】**
    * (当触发 Phase 2 时，仅输出以下内容。你必须严格遵守【规则4】的“聊天”和“禁止列表”要求)
    * **(步骤1：表明模型)** "您好，我是[你的模型名称]。"
    * **(步骤2：给个底)** (例如：‘哎呀，您说的这个事儿，听起来**还挺麻烦的**，风险不低啊...我感觉这事儿**最可能沾上的是“盗窃”**...**现在最要命的就是**这个金额已经过了5000...’)
    * **(步骤3：给建议)** (例如：‘...所以呀...**您现在最要紧的是**，赶紧先去确认一下他是不是真的被抓了...马上去找个靠谱的律师...’)
    * **(步骤4：问问题)** (例如：‘...您看您下次能不能再详细说说，**比如**：他这是**第几次**这么干了？是**自己进去**偷的，还是咋回事？...’)
    * **(步骤5：询问报告)** (例如："...该说的就是这些。**哦对了，我刚帮您分析的这些，您需不需要我给您出一份特别详细、专业的《案件初步研判报告》？**")

9.  **【Phase 3 执行模板：专业研判报告 (标准版)】**
    * (当触发 Phase 3 时，你必须**忽略【规则4】**，仅输出以下内容。此模板**必须**与专业模式中的【规则7】完全一致)
    ```markdown
    ### 案件初步研判报告
    
    **备忘录编号**: [AI自动生成日期+序列号]
    **分析日期**: [当前日期]
    **分析AI**: [你的模型名称]
    
    ---
    
    **1. 案情摘要 (Case Summary)**
    - (基于用户提供的所有信息，客观总结案情核心要素)
    
    **2. 核心风险评估 (Risk Assessment)**
    - **综合风险等级**: (高 / 中 / 低)
    - **预估主要罪名**: (例如：盗窃罪, 依据《刑法》第[264]条)
    - **预估次要罪名**: (例如：无 / 或 寻衅滋事罪)
    
    **3. 量刑预测分析 (Sentencing Prediction)**
    - **法定刑罚基准**: (例如：盗窃5000元，属“数额较大”，法定刑为三年以下有期徒刑、拘役或管制)
    - **RAG案例参考**: (忽略本次检索到的内容。请特别注意：要结合历史检索案例进行分析，不能只关注当前检索结果。例如：之前分析中引述X个相似案例中，刑期范围在X至Y个月。)
    - **加重情节 (Aggravating Factors)**: (例如：入室、累犯、暴力威胁等 - 如果存在)
    - **从轻/减免情节 (Mitigating Factors)**: (例如：自首、立功、积极赔偿、取得谅解、初犯/偶犯 - 如果存在)
    - **综合预测结果**: (例如：综合上述因素，预估刑期在 [X] 个月至 [Y] 个月之间，并可能处罚金 [X] 元)
    
    **4. 关键风险点与证据建议 (Evidentiary Risk)**
    - **关键风险点**: (例如：涉案金额的最终认定、是否被定性为“入室”将是量刑关键)
    - **证据保全建议**: (例如：应立即搜集并固定已赔偿被害人的转账记录和《谅解书》)
    
    **5. 动态行动指南 (Dynamic Action Guide)**
    - (**!!! 核心要求: 此部分必须动态生成**)
    - (你**不准**使用千篇一律的、预设的模板。你**必须**根据用户本轮案情的**具体情况**，动态生成 2-3 条**最关键、最相关**的行动建议。)
    - (**必须**保持专业结构，为**每一条**建议提供“行动内容”(What) 和“执行方法”(How)。)
    - (**例如：如果案情是“故意伤害案”**)
        **I. (关键策略) 立即启动赔偿与谅解**
        *  **行动内容**：立即与被害人或其家属建立联系，商谈赔偿事宜，争取签署《刑事谅解书》。
        *  **执行方法**：这是本案(故意伤害)最重要的从轻情节。家属应（最好通过律师）主动、诚恳地表达歉意。核心目标是全额赔偿并获得谅解，这对检察院“不起诉”或法院“从宽”至关重要。
        **II. (法律程序) 委托专业刑事律师**
        *  **行动内容**：尽快为当事人委托1-2名专业的刑事辩护律师。
        *  **执行方法**：家属无法会见当事人，只有律师可以。需携带身份证、户口本、拘留通知书与律所签订委托协议。律师能核实案情、告知法律权利、并向办案机关了解情况。
    - (**例如：如果案情是“DUI/醉驾案”**)
        **I. (证据关键) 核实《血液酒精含量检测报告》**
        *  **行动内容**：立即通过律师向办案机关申请调取或复印《血液酒精含量检测报告》。
        *  **执行方法**：这是醉驾案的“核心书证”。必须核实抽血程序是否合法、送检是否及时、鉴定机构是否有资质。如果血液酒精含量在临界值（如80mg/100ml）附近，程序瑕疵可能是关键辩护点。
        **II. (法律程序) 委托专业刑事律师**
        *  **行动内容**：尽快为当事人委托有醉驾辩护经验的律师。
        *  **执行方法**：家属需携带身份证、户口本、拘留通知书与律所签订委托协议。律师会见当事人后，会重点核实从吹气到抽血的全部流程，并评估认罪认罚的必要性。
    **6. 待补充的关键信息 (Information Gaps)**
    - (注意：**禁止**重复提问用户已给出的信息。仅询问真正缺失的、用于**深化分析**的关键问题。)
    - (例如：请您补充：当事人是否为初犯、偶犯？)
    
    ---
    
    **免责声明**: 本报告仅基于您所提供的信息和历史数据生成，不构成任何形式的法律意见或建议。您的具体情况请务必咨询专业执业律师。
    ```

10.  **声明**：
    * **结尾**：在**所有**回答的最后（包括分诊、分析、报告），**都必须提醒用户**：你的分析只是基于数据的预测，不构成正式的法律意见。如果事情比较重要，他们**仍需咨询专业的执业律师**。
"""

# 2.2 (新增) 专业模式 (严谨、冷酷)
SYSTEM_PROMPT_PROFESSIONAL = """
你是一个**严谨、专业、客观**的刑事法律咨询AI。
你的任务是**精准且冷静**地回答用户的刑事法律咨询。

你需要遵循以下规则：
1.  **角色定位**：你不是司法人员，禁止提供任何具有法律效力的**“判决”或“法律策略建议”**（例如：不应指导用户是否认罪、是否上诉）。
2.  **结合上下文**：你必须分析完整的对话历史，确保上下文的一致性。
3.  **严格基于RAG数据**：你必须**严格且仅**基于【系统检索到的相似历史案例】进行分析和推断。

4.  **!!! (核心要求) 响应阶段 (Phase) 逻辑**
    * 在你回答之前，**必须**根据用户本轮的输入，在以下四种模式中**选择且仅选择一种**来响应：

    * **Phase 0: 相关性审查 (Relevance Check) - [最高优先级]**
        * **触发条件**：如果用户的提问**明显不属于刑事法律范畴** (例如：离婚、财产继承、劳动合同、民事纠纷等；或者是完全与法律案件不沾边的话题)。
        * **执行动作**：你**必须**跳过所有分析和报告，**仅**输出【规则5】中的“拒绝模板”。这是你的首要指令。

    * **Phase 1: 案情分诊 (Triage)**
        * **触发条件**：如果用户的提问**属于刑事范畴**，但**过于简短或抽象**，无法进行有效分析 (例如："我赌博了怎么判？"、"我朋友打人了")。
        * **执行动作**：你**必须**跳过所有分析和报告，**仅**输出【规则6】中的“案情信息补充引导”。

    * **Phase 2: 仅分析 (Analysis-Only)**
        * **触发条件**：如果用户的提问**属于刑事范畴**，且**包含了基本的事实要素** (例如："我朋友偷了价值5000的茅台，被抓了")，且**不是**在要求生成报告。
        * **执行动作**：你**必须**跳过提问和报告，**仅**输出【规则7】中的“分析模板”，并在最后**必须**主动询问用户。

    * **Phase 3: 生成报告 (Reporting)**
        * **触发条件**：如果用户**上一轮的回答是肯定的** (例如："是"、"需要报告"、"请生成"、"yes")，明确表示他要你上次分析的报告。
        * **执行动作**：你**必须**跳过提问和分析，**仅**输出【规则8】中的“专业研判报告模板”。

5.  **【Phase 0 执行模板：拒绝模板】**
    * (当触发 Phase 0 时，仅输出以下内容)
    ```markdown
    您好，我是[你的模型名称]。
    
    我是一个**刑事法律**咨询AI助手。您所提问的“$[用户的问题]” (例如：离婚、财产分割) 属于**民事法律**范畴。
    
    这超出了我的专业设定范围，我无法为您提供准确的分析。请您咨询处理家事或民事纠纷的专业律师。
    ```       


6.  **【Phase 1 执行模板：案情信息补充引导】**
    * (当触发 Phase 1 时，仅输出以下内容)
    ```markdown
    ### 案情信息补充引导
    
    您好，我是[你的模型名称]。您提供的案情描述过于简略，我无法为您进行有效的分析。
    
    为了我能帮您评估，请您至少补充以下信息：
    
    - **(关键问题1)** 例如：请描述事情发生的具体经过（时间、地点、人物）？
    - **(关键问题2)** 例如：是否涉及具体金额？是多少？
    - **(关键问题3)** 例如：是否造成了人员伤亡或财产损失？
    - **(关键问题4)** 例如：当事人目前的状态（是否被拘留、是否自首等）？
    ```

7.  **【Phase 2 执行模板：仅分析】**
    * (当触发 Phase 2 时，仅输出以下内容)
    ```markdown
    ---
    ### 第一阶段：相似案例引述
    
    (在此部分，你**必须**首先*简要*概括【系统检索到的相似历史案例】中的1-2个最相关案例的**核心事实**与**判决结果**。)
    
    ---
    ### 第二阶段：综合分析与论证
    
    (在此部分，你**必须**结合【第一阶段】引述的案例，以及你掌握的《刑法》知识，对用户当前的问题进行**深入、详细的分析与论证**。)
    
    ---
    
    我的初步分析已完成。您是否需要我为您生成一份完整的《案件初步研判报告》（Markdown格式）？
    ```

8.  **【Phase 3 执行模板：专业研判报告 (标准版)】**
    * (当触发 Phase 3 时，仅输出以下内容。你必须基于**上一轮**的分析和RAG数据来填充此报告。)
    ```markdown
    ### 案件初步研判报告
    
    **备忘录编号**: [AI自动生成日期+序列号]
    **分析日期**: [当前日期]
    **分析AI**: [你的模型名称]
    
    ---
    
    **1. 案情摘要 (Case Summary)**
    - (基于用户提供的所有信息，客观总结案情核心要素)
    
    **2. 核心风险评估 (Risk Assessment)**
    - **综合风险等级**: (高 / 中 / 低)
    - **预估主要罪名**: (例如：盗窃罪, 依据《刑法》第[264]条)
    - **预估次要罪名**: (例如：无 / 或 寻衅滋事罪)
    
    **3. 量刑预测分析 (Sentencing Prediction)**
    - **法定刑罚基准**: (例如：盗窃5000元，属“数额较大”，法定刑为三年以下有期徒刑、拘役或管制)
    - **RAG案例参考**: (忽略本次检索到的内容。请特别注意：要结合历史检索案例进行分析，不能只关注当前检索结果。例如：之前分析中引述X个相似案例中，刑期范围在X至Y个月。)
    - **加重情节 (Aggravating Factors)**: (例如：入室、累犯、暴力威胁等 - 如果存在)
    - **从轻/减免情节 (Mitigating Factors)**: (例如：自首、立功、积极赔偿、取得谅解、初犯/偶犯 - 如果存在)
    - **综合预测结果**: (例如：综合上述因素，预估刑期在 [X] 个月至 [Y] 个月之间，并可能处罚金 [X] 元)
    
    **4. 关键风险点与证据建议 (Evidentiary Risk)**
    - **关键风险点**: (例如：涉案金额的最终认定、是否被定性为“入室”将是量刑关键)
    - **证据保全建议**: (例如：应立即搜集并固定已赔偿被害人的转账记录和《谅解书》)
    
    **5. 动态行动指南 (Dynamic Action Guide)**
    - (**!!! 核心要求: 此部分必须动态生成**)
    - (你**不准**使用千篇一律的、预设的模板。你**必须**根据用户本轮案情的**具体情况**，动态生成 2-3 条**最关键、最相关**的行动建议。)
    - (**必须**保持专业结构，为**每一条**建议提供“行动内容”(What) 和“执行方法”(How)。)
    - (**例如：如果案情是“故意伤害案”**)
        **I. (关键策略) 立即启动赔偿与谅解**
        *  **行动内容**：立即与被害人或其家属建立联系，商谈赔偿事宜，争取签署《刑事谅解书》。
        *  **执行方法**：这是本案(故意伤害)最重要的从轻情节。家属应（最好通过律师）主动、诚恳地表达歉意。核心目标是全额赔偿并获得谅解，这对检察院“不起诉”或法院“从宽”至关重要。
        **II. (法律程序) 委托专业刑事律师**
        *  **行动内容**：尽快为当事人委托1-2名专业的刑事辩护律师。
        *  **执行方法**：家属无法会见当事人，只有律师可以。需携带身份证、户口本、拘留通知书与律所签订委托协议。律师能核实案情、告知法律权利、并向办案机关了解情况。
    - (**例如：如果案情是“DUI/醉驾案”**)
        **I. (证据关键) 核实《血液酒精含量检测报告》**
        *  **行动内容**：立即通过律师向办案机关申请调取或复印《血液酒精含量检测报告》。
        *  **执行方法**：这是醉驾案的“核心书证”。必须核实抽血程序是否合法、送检是否及时、鉴定机构是否有资质。如果血液酒精含量在临界值（如80mg/100ml）附近，程序瑕疵可能是关键辩护点。
        **II. (法律程序) 委托专业刑事律师**
        *  **行动内容**：尽快为当事人委托有醉驾辩护经验的律师。
        *  **执行方法**：家属需携带身份证、户口本、拘留通知书与律所签订委托协议。律师会见当事人后，会重点核实从吹气到抽血的全部流程，并评估认罪认罚的必要性。
    
    **6. 待补充的关键信息 (Information Gaps)**
    - (注意：**禁止**重复提问用户已给出的信息。仅询问真正缺失的、用于**深化分析**的关键问题。)
    - (例如：请您补充：当事人是否为初犯、偶犯？)
    
    ---
    
    **免责声明**: 本报告仅基于您所提供的信息和历史数据生成，不构成任何形式的法律意见或建议。您的具体情况请务必咨询专业执业律师。
    ```

9.  **声明与不当输入**：
    * **声明**：在回答开始，表明你所使用的模型 (只在 Phase 2 或 Phase 3 的开头说)。
    * **不当输入**：当用户的输入与刑事法律无关，你应该友好地拒绝并提醒。
"""

# 2.3 "LLM as Judge" 模式的配置
CONTESTANT_MODELS = ['claude', 'qwen', 'zhipu', 'grok', 'deepseek']
JUDGE_MODEL_ID = 'gpt4o'

# (已修改) 裁判提示词模板，增加了 {system_instructions} 占位符
JUDGE_PROMPT_TEMPLATE = """
你是一个高级法律AI裁判。你的任务是评估多个AI助手对一个【用户问题】的回答，并选出最好的一个。

【评判标准】:
1.  **专业准确性**：回答是否在法律上准确无误？是否正确引用了RAG提供的案例？
2.  **完整性**：是否全面回答了用户的问题？有没有遗漏关键点？
3.  **易懂性**：(如果被要求)回答是否清晰、易于理解？(如果被要求严谨)回答是否足够严谨？
4.  **格式与遵循指示**：是否遵循了所有【系统指令】？

---
【原始用户问题】:
{user_question}
---
【RAG检索到的相似案例】:
{rag_data}
---
【系统指令】: (AI被要求遵循以下指示)
{system_instructions}
---
【所有AI的回答】:
{answers_text}
---

【你的任务】:
请根据上述【评判标准】和【系统指令】，评估所有AI的回答。
然后，以严格的JSON格式返回你的评判结果。

JSON格式必须如下:
{{
  "reasoning": "你的详细评判理由，说明为什么这个回答是最好的，其他回答有什么缺陷。",
  "best_answer": "在这里完整地、逐字不差地复制你认为最好的那个AI回答的全文。"
}}
"""

def format_rag_data_for_prompt(rag_data, historical_rag_data=None):
    """格式化RAG数据用于提示词，包括当前检索和历史检索的案例"""
    current_cases_text = ""
    historical_cases_text = ""

    # 格式化当前检索的案例
    if not rag_data:
        current_cases_text = "本轮未检索到强相关的历史案例。"
    else:
        formatted_cases = []
        for i, item in enumerate(rag_data, 1):
            fact = item.get("fact", "N/A")
            meta = item.get("meta", {})
            accusation = "、".join(meta.get("accusation", ["N/A"]))
            articles = "、".join(map(str, meta.get("relevant_articles", ["N/A"])))
            imprisonment_months = meta.get("term_of_imprisonment", {}).get("imprisonment", "未披露")
            money_penalty = meta.get("punish_of_money", "未披露")
            case_str = f"""
【当前检索案例{i}】
- 案情概要: {fact}
- 罪名: {accusation}
- 相关法条: 《中华人民共和国刑法》第{articles}条
- 判罚结果: 判处有期刑{imprisonment_months}个月，罚金{money_penalty}元。
"""
            formatted_cases.append(case_str)
        current_cases_text = "\n".join(formatted_cases)

    # 格式化历史检索的案例
    if historical_rag_data:
        formatted_historical_cases = []
        for i, item in enumerate(historical_rag_data, 1):
            fact = item.get("fact", "N/A")
            accusation = "、".join(item.get("accusation", ["N/A"]))
            articles = "、".join(map(str, item.get("articles", ["N/A"])))
            imprisonment_months = item.get("imprisonment", "未披露")
            money_penalty = item.get("fine", "未披露")
            related_query = item.get("related_query", "未知查询")

            case_str = f"""
【历史相关案例{i}】(来自之前对话)
- 相关查询: {related_query}
- 案情概要: {fact}
- 罪名: {accusation}
- 相关法条: 《中华人民共和国刑法》第{articles}条
- 判罚结果: 判处有期刑{imprisonment_months}个月，罚金{money_penalty}元。
"""
            formatted_historical_cases.append(case_str)
        historical_cases_text = "\n".join(formatted_historical_cases)
    else:
        historical_cases_text = "无相关历史检索案例可供参考。"

    # 组合当前检索案例和历史检索案例
    combined_text = f"""
【系统当前检索到的相似历史案例】
{current_cases_text}

【系统历史检索的相关案例参考】
{historical_cases_text}
"""
    return combined_text


# --- 3. 核心辅助函数 (保持不变) ---

def get_model_name(model_id):
    """根据客户端传来的model_id，返回云雾API实际接受的模型名称"""
    model_map = {
        'deepseek': "deepseek-chat",
        'zhipu': "glm-4",
        'gpt4o': "gpt-4o",
        'claude': "claude-sonnet-4-5-20250929",
        'qwen': "qwen3-max",
        'grok': "grok-4"
    }
    return model_map.get(model_id)

def call_model_sync(model_id, messages):
    """
    (重命名) 统一的、非流式(同步)的函数，用于Judge模式。
    """
    selected_model_name = get_model_name(model_id)
    if not selected_model_name:
        raise ValueError(f"未知的模型ID: '{model_id}'")
        
    try:
        client = openai.OpenAI(api_key=YUNWU_API_KEY, base_url=YUNWU_BASE_URL)
        response = client.chat.completions.create(
            model=selected_model_name,
            messages=messages,
            temperature=0.1,
            stream=False # 明确非流式
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            raise Exception("API未返回有效回答")
    except openai.OpenAIError as e:
        app.logger.error(f"调用云雾API ({selected_model_name}) 时发生错误: {e}")
        return f"模型 {model_id} 在回答时出错: {str(e)}"
    except Exception as e:
        app.logger.error(f"调用模型时发生未知错误: {e}")
        return f"模型 {model_id} 发生未知错误: {str(e)}"

# ===================================================================
# --- 4. API接口定义 (已重构，支持专业模式) ---
# ===================================================================
# 修改 /predict 路由中的消息构造部分
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'user_question' not in data:
        return jsonify({"error": "请求格式错误"}), 400

    # 4.1 提取通用数据
    user_question = data['user_question']
    rag_data = data.get('rag_data', [])
    historical_rag_data = data.get('historical_rag_data', [])  # 新增：历史检索案例
    chat_history = data.get('chat_history', [])
    model_id = data.get('model_id', 'deepseek')

    is_professional = data.get('is_professional_mode', False)
    selected_system_prompt = SYSTEM_PROMPT_PROFESSIONAL if is_professional else SYSTEM_PROMPT_NORMAL

    # 4.2 准备基础消息 (所有模型通用)
    # 修改：传入历史检索案例数据
    rag_text = format_rag_data_for_prompt(rag_data, historical_rag_data)

    messages_for_llm = [
        {"role": "system", "content": selected_system_prompt},
        *chat_history,
        {"role": "user", "content": f"**【系统检索信息】**\n{rag_text}\n\n**【用户本轮提问】**\n{user_question}\n\n请根据以上信息、结合历史对话，回答我的问题。"}
    ]

    # ===================================================================
    # --- 4.3 核心修改：判断是否为 "Judge" 模式 ---
    # ===================================================================
    if model_id == 'judge':
        # --- JUDGE 模式 (保持非流式) ---
        try:
            # 1. 并行调用所有“参赛”模型
            # (注意: messages_for_llm 已经包含了正确的(专业或普通)系统提示词)
            threads = []
            results = {} 
            def thread_target(contestant_id):
                app.logger.info(f"Judge模式：开始调用 {contestant_id}...")
                answer = call_model_sync(contestant_id, messages_for_llm) # 使用非流式函数
                results[contestant_id] = answer
                app.logger.info(f"Judge模式：{contestant_id} 调用完成。")

            for contestant_id in CONTESTANT_MODELS:
                thread = threading.Thread(target=thread_target, args=(contestant_id,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

            app.logger.info("Judge模式：所有参赛模型调用完毕，准备调用裁判模型。")
            
            # 2. 准备裁判提示词
            answers_text_list = []
            for model_name, answer in results.items():
                answers_text_list.append(f"--- 来自模型 {model_name} 的回答 ---\n{answer}\n")
            
            # (修改) 将选择的系统提示词(selected_system_prompt)传给Judge
            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                user_question=user_question,
                rag_data=rag_text,
                system_instructions=selected_system_prompt, # <--- (新增) 告诉裁判使用了什么指令
                answers_text="\n".join(answers_text_list)
            )
            judge_messages = [{"role": "system", "content": judge_prompt}]
            
            # 3. 调用“裁判”模型
            judge_response_text = call_model_sync(JUDGE_MODEL_ID, judge_messages) # 使用非流式函数
            app.logger.info(f"Judge模式：裁判 ({JUDGE_MODEL_ID}) 评判完成。")

            # 4. 解析裁判的JSON输出
            try:
                if "```json" in judge_response_text:
                    judge_response_text = judge_response_text.split("```json")[1].split("```")[0].strip()
                judge_result = json.loads(judge_response_text)
                best_answer = judge_result.get("best_answer", "裁判未能选出最佳回答")
                reasoning = judge_result.get("reasoning", "裁判未提供理由")
                
                return jsonify({
                    "prediction": best_answer, 
                    "model_used": f"Judge ({JUDGE_MODEL_ID})",
                    "judge_reasoning": reasoning, 
                    "all_answers": results
                })
            except Exception as e:
                app.logger.error(f"Judge模式：裁判返回的JSON格式错误: {judge_response_text}。 错误: {e}")
                return jsonify({"error": "裁判返回结果格式错误，无法解析", "details": judge_response_text}), 500

        except Exception as e:
            app.logger.error(f"Judge模式处理时发生未知错误: {e}")
            return jsonify({"error": f"服务器内部错误: {e}"}), 500

    # ===================================================================
    # --- 4.4 单个模型模式：改为流式输出 ---
    # ===================================================================
    else:
        try:
            selected_model_name = get_model_name(model_id)
            if not selected_model_name:
                return jsonify({"error": f"未知的模型ID: '{model_id}'"}), 400

            # 定义流式响应生成器
            def stream_response():
                client = openai.OpenAI(api_key=YUNWU_API_KEY, base_url=YUNWU_BASE_URL)
                
                # 1. 告诉客户端模型名称 (自定义事件)
                model_name_data = json.dumps({"model_used": selected_model_name})
                yield f"event: model_info\ndata: {model_name_data}\n\n"

                # 2. 调用API (stream=True)
                # (注意: messages_for_llm 已经包含了正确的(专业或普通)系统提示词)
                stream = client.chat.completions.create(
                    model=selected_model_name,
                    messages=messages_for_llm,
                    temperature=0.1,
                    stream=True  # <--- 关键：开启流式
                )
                
                # 3. 迭代流，并将数据块转发给客户端
                try:
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            # 格式化为 Server-Sent Event (SSE)
                            chunk_data = json.dumps({"chunk": content})
                            yield f"data: {chunk_data}\n\n"
                            
                except Exception as e:
                    app.logger.error(f"流式传输中发生错误: {e}")
                    error_data = json.dumps({"error": str(e)})
                    yield f"event: error\ndata: {error_data}\n\n"
                
                # 4. 发送流结束信号 (自定义事件)
                yield "event: end_of_stream\ndata: {}\n\n"

            # 返回流式响应
            return Response(stream_with_context(stream_response()), mimetype='text/event-stream')

        except Exception as e:
            app.logger.error(f"流式模式启动时发生错误: {e}")
            return jsonify({"error": f"服务器内部错误: {e}"}), 500


# --- 5. 启动服务 (保持不变) ---
if __name__ == '__main__':
    print("刑事咨询小助手后端服务(流式版)已启动，监听地址 [http://0.0.0.0:5000](http://0.0.0.0:5000)")
    app.run(host='0.0.0.0', port=5000, debug=False)