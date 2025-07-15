import os
import datetime
import json
from dotenv import load_dotenv
import markdown
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from email_sender import schedule_email
import re
import requests
from urllib.parse import urlparse
import time

# 加载环境变量
load_dotenv()

# 配置API密钥
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# 确保API基础URL格式正确
if DEEPSEEK_API_BASE and not DEEPSEEK_API_BASE.endswith("/v1"):
    DEEPSEEK_API_BASE = f"{DEEPSEEK_API_BASE}/v1"

# 创建 DeepSeek LLM 配置
deepseek_llm = LLM(
    model="deepseek-chat",
    base_url=DEEPSEEK_API_BASE,
    api_key=DEEPSEEK_API_KEY
)

# 验证配置
if not DEEPSEEK_API_KEY:
    print("警告: 未找到DeepSeek API密钥。请在.env文件中设置DEEPSEEK_API_KEY。")
if not SERPER_API_KEY:
    print("警告: 未找到SERPER_API_KEY。请在.env文件中设置SERPER_API_KEY。")

# 新能源汽车关键词列表
EV_KEYWORDS = [
    "新能源汽车", "电动汽车", "纯电动", "混合动力", "插电混动", "氢燃料", "电池技术", 
    "充电桩", "充电设施", "续航里程", "电池寿命", "特斯拉", "比亚迪", "蔚来", "小鹏", 
    "理想", "长安汽车", "新能源政策", "电动汽车补贴", "碳中和", "碳排放", "动力电池", 
    "固态电池", "电池回收", "新能源汽车销量", "电动汽车技术",
    # 新增加的关键词
    "深蓝", "阿维塔", "埃安", "吉利银河", "吉利几何", "零跑", "欧拉", "方程豹", 
    "飞凡", "奇瑞新能源", "腾势", "传统燃油车", "东风纳米", "ARCFOX 极狐", "哪吒", 
    "人形机器人", "具身智能", "FSD", "激光雷达", "飞行汽车","启源","长安深蓝","长安阿维塔","问界","赛力斯","岚图","智己","上汽大众","上汽通用","上汽通用五菱","上汽奥迪","上汽名爵","上汽荣威","上汽大通","上汽通用别克","上汽通用雪佛兰","上汽通用福特","上汽通用本田","上汽通用丰田","上汽通用日产","上汽通用现代","上汽通用起亚","上汽通用雪佛兰","上汽通用福特","上汽通用本田","上汽通用丰田","上汽通用日产","上汽通用现代","上汽通用起亚"
    # 新增全球品牌与技术关键词
    "Volkswagen ID", "Toyota bZ4X", "Nissan Ariya", "Hyundai Ioniq", "Kia EV6",
    "General Motors Ultium", "Ford Mustang Mach-E", "Rivian", "Lucid", "Polestar",
    "Solid-state battery", "Battery swapping", "Vehicle-to-grid", "Fast charging",
    "EV supply chain", "Battery recycling"
]

# 创建CrewAI代理
def create_crew_agents():
    # 创建搜索工具 - 配置为返回更多结果且支持全球搜索
    search_tool = SerperDevTool(
        n_results=20  # 增加返回结果数量
    )
    
    # 创建针对特定地区的搜索工具
    search_tool_china = SerperDevTool(
        country="cn",
        locale="zh",
        n_results=20
    )
    
    search_tool_usa = SerperDevTool(
        country="us",
        locale="en",
        n_results=20
    )
    
    search_tool_europe = SerperDevTool(
        country="de",  # 德国
        locale="de",
        n_results=20
    )
    
    search_tool_japan = SerperDevTool(
        country="jp",
        locale="ja",
        n_results=20
    )
    
    # 内容收集代理
    content_collector = Agent(
        role="新能源汽车内容收集专家",
        goal="收集和整理全球新能源汽车相关的最新信息",
        backstory="""你是一位资深的新能源汽车行业研究员，擅长从海量信息中收集和整理有价值的内容。
        你专注于全球新能源汽车行业的最新动态，包括各大汽车厂商信息、政策变化、技术突破、市场趋势和消费者偏好等。
        你特别关注长安汽车及其新兴品牌如深蓝、阿维塔等的动态，以及人形机器人、激光雷达等前沿技术的进展。""",
        verbose=True,
        llm=deepseek_llm,  # 使用DeepSeek LLM
        tools=[search_tool, search_tool_china, search_tool_usa, search_tool_europe, search_tool_japan]
    )
    
    # 内容分析代理
    content_analyst = Agent(
        role="新能源汽车行业分析专家",
        goal="深入分析收集到的新能源汽车内容，识别关键趋势、市场变化和技术进展，提供有洞察力的行业解读",
        backstory="""你是一位经验丰富的新能源汽车行业分析师，拥有深厚的行业知识和敏锐的洞察力。
        你善于解读数据背后的趋势，分析技术路线的优劣，预测市场变化的方向。
        你特别擅长分析新能源汽车行业的竞争格局、技术路线、政策影响和消费者行为等方面。""",
        verbose=True,
        allow_delegation=True,
        llm=deepseek_llm,  # 使用DeepSeek LLM
        tools=[search_tool, search_tool_china, search_tool_usa, search_tool_europe, search_tool_japan]
    )
    
    # 报告撰写代理
    report_writer = Agent(
        role="新能源汽车分析报告撰写专家",
        goal="创建高质量、严谨专业且结构清晰的新能源汽车行业分析报告",
        backstory="""你是一位资深的专业报告撰写专家，擅长将复杂的行业信息转化为逻辑严密、数据准确的高水平分析报告。
        你的报告风格严肃正式，措辞精准，论证有力，避免使用口语化表达和夸张修饰。
        你坚持学术规范，对每个重要论点和数据都提供相应的信息来源和引用链接。
        你熟悉专业研究报告的撰写标准，能够创建既有学术价值又有实用参考意义的行业分析文档。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm  # 使用DeepSeek LLM
    )
    
    return content_collector, content_analyst, report_writer

# 创建任务
def create_crew_tasks(keywords, time_range_days, content_collector, content_analyst, report_writer):
    # 计算日期范围
    today = datetime.datetime.now()
    start_date = today - datetime.timedelta(days=time_range_days)
    date_range = f"after:{start_date.strftime('%Y-%m-%d')}"
    
    # 内容收集任务
    collect_task = Task(
        description=f"""
        搜集与"{keywords or '新能源汽车'}"相关的最新全球内容，重点关注最近{time_range_days}天内的信息。
        使用日期范围：{date_range}（从{start_date.strftime('%Y-%m-%d')}到{today.strftime('%Y-%m-%d')}）
        
        【重要】必须只收集实际存在的内容，严禁创造模拟数据或未来内容：
        - 只收集真实存在、可验证的信息
        - 不要创建未来日期的内容或引用
        - 不要生成看起来真实但实际不存在的URL
        - 所有引用必须可以通过浏览器实际访问
        
        你有多个搜索工具可用，每个工具针对不同地区优化：
        - 第一个工具(search_tool)：全球通用搜索
        - 第二个工具(search_tool_china)：专门搜索中国内容
        - 第三个工具(search_tool_usa)：专门搜索美国内容
        - 第四个工具(search_tool_europe)：专门搜索欧洲内容
        - 第五个工具(search_tool_japan)：专门搜索日本内容
        
        使用搜索工具时，请使用格式：
        tool.run(search_query="你的搜索关键词")
        例如: tool.run(search_query="新能源汽车 最新发展 {date_range}")
        
        为了获取全球视角，请确保使用所有工具搜索相关内容。对于中国市场内容，可以使用中文关键词；对于其他国家市场，使用英文或当地语言关键词。
        
        搜索内容包括但不限于：
        1. 主要汽车厂商（比亚迪、特斯拉、长安等）的新闻和动态
        2. 新能源汽车技术进展（电池技术、智能驾驶等）
        3. 各国新能源汽车政策变化
        4. 市场销量和消费者趋势数据
        5. 创新技术和前沿研究（固态电池、飞行汽车等）
        
        对于每个信息源，必须详细提供：
        - 标题和来源
        - 发布日期（必须是真实日期，不要创造未来日期）
        - 主要内容摘要
        - 完整的URL地址（这一点极其重要，后续报告需要引用）
          * 必须提供完整的文章URL，而非网站首页
          * 确保URL格式正确，以http://或https://开头
          * 链接到具体的新闻文章或报告，而非搜索结果页面
          * 如果原始URL很长，不要截断它
          * 所有URL必须真实存在，可通过浏览器访问
        - 来源的可信度评估
        - 内容所属的地区/国家
        
        特别关注：
        - 长安汽车及其品牌（深蓝、阿维塔等）的相关信息
        - 电池技术的最新进展
        - 智能驾驶和自动驾驶的发展
        - 新能源汽车政策变化
        - 各国市场比较数据
        
        搜集至少40-50条相关内容，确保内容的多样性和全球覆盖面。每条内容必须有明确的来源URL，以便后续分析和报告引用。

        【URL收集指南】
        1. 确保每个引用的URL都直接指向原始文章页面
        2. 避免使用短链接或重定向链接
        3. 验证URL是否可以直接访问（不要使用需要登录的内容）
        4. 如果原始来源有付费墙，尝试找到类似内容的免费替代来源
        5. 对于非英文/中文内容，尽量找到原始语言的官方来源
        6. 只收集实际存在的URL，不要创建看似真实但实际不存在的URL
        """,
        agent=content_collector,
        expected_output="一份包含40-50条来自全球各地的新能源汽车最新内容的详细清单，包括标题、来源、日期、内容摘要、完整URL地址和地区信息。URL必须是具体文章的直接链接，而非网站首页，且必须是实际存在的真实URL。"
    )
    
    # 内容分析任务
    analyze_task = Task(
        description=f"""
        基于收集到的新能源汽车相关内容，进行深入分析，识别最近{time_range_days}天内的关键趋势和洞察。
        
        【重要】严禁使用模拟数据或创造内容：
        - 分析必须基于实际收集到的内容，不得添加虚构内容
        - 不得创建未来日期的数据或引用
        - 所有引用必须基于真实存在的URL
        - 如果某方面缺乏足够数据支持，应明确说明，而非创造数据
        
        你有多个搜索工具可用，每个工具针对不同地区优化：
        - 第一个工具(search_tool)：全球通用搜索
        - 第二个工具(search_tool_china)：专门搜索中国内容
        - 第三个工具(search_tool_usa)：专门搜索美国内容
        - 第四个工具(search_tool_europe)：专门搜索欧洲内容
        - 第五个工具(search_tool_japan)：专门搜索日本内容
        
        使用搜索工具时，请使用格式：
        tool.run(search_query="你的搜索关键词")
        
        分析内容应包括：
        1. 全球新能源汽车行业整体发展趋势对比分析（中国、美国、欧洲、日本等主要市场）
        2. 各国主要汽车厂商的最新动向和战略变化
        3. 不同地区消费者购买决策的变化因素和偏好差异
        4. 各国新能源汽车政策的影响和方向
        5. 电池技术、智能驾驶等核心技术的最新进展
        6. 新能源汽车与传统燃油车的市场竞争态势
        7. 长安汽车及其品牌（深蓝、阿维塔等）的市场表现和技术优势
        8. 人形机器人、激光雷达等前沿技术在汽车领域的应用进展
        9. 全球电池供应链分析
        10. 各国充电基础设施建设比较
        
        分析要求：
        - 确保分析观点与数据有明确的信息来源，保留原始内容的URL引用
        - 对重要数据和结论必须标明来源出处
        - 遵循学术严谨性，保持客观公正的分析态度
        - 分析要有深度，不仅描述现象，更要探究背后的原因和未来趋势
        - 对比分析不同来源的信息，确保多角度审视问题
        - 适当引用权威机构和专家的观点，增强分析的可信度
        - 注重国际比较分析，探讨各国市场差异及其原因
        - 严禁创造不存在的数据或来源，只分析实际搜集到的内容
        
        提供有充分依据支持的深入分析和见解，确保分析内容全面、客观、专业，且具有可溯源性。
        """,
        agent=content_analyst,
        expected_output="一份全面的新能源汽车行业全球比较分析报告，包含各地区关键趋势、市场动态、技术进展和战略洞察，并附有完整信息来源引用。分析仅基于真实数据，不包含模拟或虚构内容。",
        context=[collect_task]
    )
    
    # 报告撰写任务
    write_report_task = Task(
        description=f"""
        根据已收集的内容和分析结果，创建一份严谨、专业的新能源汽车行业全球分析报告。
        
        报告要点：
        1. 报告标题应为"新能源汽车全球分析报告"，不要包含具体年份
        2. 在标题下方明确标示：
           **数据截止日期：{today.strftime('%Y年%m月%d日')} | 分析时间范围：最近{time_range_days}天**
        
        【重要】直接编写Markdown内容，不要添加```markdown或```等代码块标记。
        
        【禁止使用模拟数据】
        - 严禁使用任何模拟或假设的数据，只能使用实际存在的数据
        - 严禁生成虚构的日期、数据、网址或引用
        - 严禁创建未来日期的内容或引用
        - 只使用真实存在的URL，可通过浏览器访问的内容
        - 对于没有明确数据支持的内容，应明确标注为趋势预测或行业展望，而非确定性结论
        
        报告必须使用Markdown格式，包含以下部分：
        1. 标题和执行摘要
        2. 全球新能源汽车市场概况
           - 中国市场
           - 美国市场
           - 欧洲市场
           - 日本市场
           - 其他新兴市场
        3. 各主要厂商竞争格局分析
           - 中国厂商（比亚迪、长安、蔚来等）
           - 美国厂商（特斯拉、通用、福特等）
           - 欧洲厂商（大众、宝马、奔驰等）
           - 日本厂商（丰田、本田、日产等）
        4. 消费者行为洞察（各地区差异对比）
        5. 各国政策环境分析
        6. 全球技术发展趋势
        7. 长安汽车及其品牌专题分析
        8. 前沿技术全球发展展望
        9. 战略建议
        10. 附录（数据来源、术语解释等）
        
        报告要求：
        - 使用严谨、正式的学术语言，避免口语化表达
        - 对所有重要观点和数据必须提供原文引用的URL地址，格式为：[内容描述](URL)
          * URL必须是完整的具体文章链接，不能仅链接到首页
          * 确保每个URL都能直接访问到引用的具体内容
          * 引用URL时不要修改或截断原始URL
          * 对于重要数据或关键观点，至少提供2-3个不同来源的URL引用
          * 严禁创建或引用不存在的URL，特别是未来日期的URL
        - 在每个章节末尾整理相关参考资料，格式为[标题](URL链接)
        - 适当使用表格呈现各国数据对比
        - 报告结构严谨，逻辑清晰，论证有力
        - 所有内容必须基于事实和数据，避免无根据的推测
        - 不要在报告结尾添加标准声明或遵循的学术标准列表
        
        【URL引用规范】
        - 确保所有URL都是以http://或https://开头的完整链接
        - 引用格式必须为Markdown标准格式：[锚文本](URL)
        - 每个URL都应该链接到具体内容，而非网站首页
        - 不要使用相对路径，必须使用绝对URL
        - 如果原始URL包含中文或特殊字符，确保正确编码
        - 检查最终报告中的所有URL格式是否正确，是否可以直接点击访问
        - 只使用当前实际存在的URL，不要创建未来日期的URL或不存在的页面
        """,
        agent=report_writer,
        expected_output="一份专业严谨、包含全球视角和完整引用链接的Markdown格式新能源汽车行业分析报告，标题不含年份，结尾无标准声明，不包含代码块标记。所有引用URL均为完整的具体文章链接，且必须是实际存在的真实数据。",
        context=[analyze_task]
    )
    
    return collect_task, analyze_task, write_report_task

# 运行新能源汽车内容分析
def run_ev_news_analysis(keywords=None, email=None, time_range_days=30, send_time="now", ensure_chinese=False):
    """
    使用CrewAI模型和代理运行新能源汽车内容分析（全球范围）
    
    参数:
    - keywords: 搜索关键词(可选，默认使用新能源汽车相关关键词)
    - email: 报告接收邮箱
    - time_range_days: 内容搜索时间范围（天）
    - send_time: 报告发送时间（"now"为立即发送，或datetime对象）
    - ensure_chinese: 确保报告以中文呈现（默认False）
    
    返回:
    - 报告内容和状态信息的字典
    """
    # 创建代理
    content_collector, content_analyst, report_writer = create_crew_agents()
    
    # 创建任务
    collect_task, analyze_task, write_report_task = create_crew_tasks(
        keywords, time_range_days, content_collector, content_analyst, report_writer
    )
    
    # 创建Crew并指定使用DeepSeek LLM
    ev_analysis_crew = Crew(
        agents=[content_collector, content_analyst, report_writer],
        tasks=[collect_task, analyze_task, write_report_task],
        verbose=True,
        process=Process.sequential,  # 按顺序执行任务
        llm=deepseek_llm  # 使用DeepSeek LLM
    )
    
    print(f"开始分析关键词: {keywords or '新能源汽车'}, 时间范围: 最近{time_range_days}天")
    
    # 运行Crew执行任务
    report_result = ev_analysis_crew.kickoff()
    
    # CrewOutput 对象处理
    if hasattr(report_result, 'raw'):
        report_content = report_result.raw
    elif hasattr(report_result, 'output'):
        report_content = report_result.output
    elif isinstance(report_result, str):
        report_content = report_result
    else:
        report_content = str(report_result)
    
    # 清理报告内容，移除可能的markdown代码块标记
    # 移除开头的```markdown或```等标记
    report_content = report_content.strip()
    
    # 处理开头可能有多个反引号的情况（如````markdown）
    if report_content.startswith("`"):
        # 找到第一行
        first_line_end = report_content.find("\n")
        if first_line_end > 0:
            first_line = report_content[:first_line_end].strip()
            # 如果第一行只包含反引号和可能的语言标识，则移除整行
            if all(c == '`' or c.isalpha() or c.isspace() for c in first_line):
                report_content = report_content[first_line_end+1:].strip()
    
    # 处理标准```markdown```情况
    if report_content.startswith("```"):
        # 查找第一个代码块结束标记
        first_block_end = report_content.find("```", 3)
        if first_block_end > 0:
            # 如果是单独的代码块标记行，直接跳过这一行
            line_end = report_content.find("\n", 3)
            if line_end > 0 and line_end < first_block_end:
                report_content = report_content[line_end+1:].strip()
            else:
                # 否则可能是整个内容被包装在代码块中
                report_content = report_content[first_block_end+3:].strip()
    
    # 处理可能存在的嵌套代码块
    while report_content.startswith("```") and "```" in report_content[3:]:
        # 查找第一个代码块结束标记
        first_block_end = report_content.find("```", 3)
        if first_block_end > 0:
            report_content = report_content[first_block_end+3:].strip()
        else:
            break
    
    # 移除末尾的```标记
    if report_content.endswith("```"):
        report_content = report_content[:report_content.rfind("```")].strip()
    
    # 再次检查末尾是否有多个反引号
    while report_content.endswith("`"):
        report_content = report_content.rstrip("`").strip()
    
    # URL格式检查与修复
    print("检查并修复报告中的URL...")
    # 查找Markdown格式的URL链接
    url_pattern = re.compile(r'\[(.*?)\]\((.*?)\)')
    
    # 检查报告中的URL格式
    def check_and_fix_urls(content):
        # 导入必要的库
        import requests
        from urllib.parse import urlparse
        import time
        
        # 设置请求头，模拟浏览器行为
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 查找所有URL链接
        matches = url_pattern.findall(content)
        fixed_content = content
        url_fixes_count = 0
        
        for text, url in matches:
            fixed_url = url
            need_replacement = False
            
            # 1. 检查并修复URL协议（确保使用HTTPS）
            if not url.startswith('http://') and not url.startswith('https://'):
                # 如果URL不完整，添加HTTPS
                fixed_url = f"https://{url}" if not url.startswith('//') else f"https:{url}"
                need_replacement = True
                print(f"修复URL协议: {url} -> {fixed_url}")
                url_fixes_count += 1
            elif url.startswith('http://'):
                # 将HTTP转换为HTTPS
                fixed_url = url.replace('http://', 'https://')
                need_replacement = True
                print(f"HTTP转HTTPS: {url} -> {fixed_url}")
                url_fixes_count += 1
            
            # 2. 检查URL是否可能是首页
            parsed_url = urlparse(fixed_url)
            path_parts = parsed_url.path.strip('/').split('/')
            homepage_indicators = [
                len(path_parts) < 2 and not parsed_url.query,  # URL路径过短且无查询参数
                fixed_url.endswith(('.com', '.cn', '.org', '.net')) and not parsed_url.path.strip('/'),  # 直接以域名结尾
                '/index.' in fixed_url or '/home.' in fixed_url  # 包含index或home
            ]
            
            if any(homepage_indicators):
                print(f"警告: URL可能是首页而非具体文章: {fixed_url}")
            
            # 3. 检查URL可访问性（仅检测，不替换）
            try:
                # 对每个URL尝试进行HEAD请求，检查状态码
                response = requests.head(fixed_url, headers=headers, timeout=5, allow_redirects=True)
                
                # 检查状态码是否表示错误
                if response.status_code >= 400:
                    print(f"警告: URL可能不可访问 (状态码: {response.status_code}): {fixed_url}")
                    print(f"信息: 保留原始URL，不替换为域名首页。如果该URL确实有效，请检查SERPER搜索结果。")
                
                # 限制请求速率，避免被目标网站封锁
                time.sleep(0.2)
            except Exception as e:
                # 请求失败也只提供警告，不修改URL
                print(f"警告: 无法验证URL (原因: {str(e)}): {fixed_url}")
                print(f"信息: 保留原始URL，不替换为域名首页。这可能是临时网络问题或安全限制。")
            
            # 只应用URL格式修复（如果需要），不进行内容替换
            if need_replacement:
                fixed_content = fixed_content.replace(f"[{text}]({url})", f"[{text}]({fixed_url})")
        
        print(f"完成URL检查，修复了 {url_fixes_count} 个URL格式问题。保留所有原始URL引用，包括可能存在访问问题的链接。")
        return fixed_content
    
    # 应用URL检查与修复
    report_content = check_and_fix_urls(report_content)
    
    # 保存报告为Markdown文件
    report_filename = f"新能源汽车全球分析报告_{datetime.datetime.now().strftime('%Y%m%d')}.md"
    
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"报告已生成: {report_filename}")
    
    # 如果提供了邮箱，则发送邮件
    send_status = None
    if email:
        # 获取邮箱密码
        email_password = os.environ.get("EMAIL_PASSWORD", "")
        
        if send_time == "now":
            # 立即发送
            try:
                # 使用Markdown邮件正文，并确保完整保留Markdown格式
                email_body = report_content
                
                send_status = schedule_email(
                    recipient=email,
                    subject="新能源汽车全球分析报告",
                    body=email_body,
                    content_type="markdown",  # 指定内容类型为Markdown
                    attachment_path=report_filename,
                    password=email_password
                )
                print(f"邮件发送状态: {send_status}")
            except Exception as e:
                send_status = f"邮件发送失败: {str(e)}"
                print(send_status)
        else:
            # 计划发送
            try:
                # 使用Markdown邮件正文，并确保完整保留Markdown格式
                email_body = report_content
                
                send_status = "邮件将在指定时间发送"
                # 将邮件发送任务添加到计划
                schedule_email(
                    recipient=email,
                    subject="新能源汽车全球分析报告",
                    body=email_body,
                    content_type="markdown",  # 指定内容类型为Markdown
                    attachment_path=report_filename,
                    send_time=send_time,
                    password=email_password
                )
                print(f"邮件计划发送状态: {send_status}")
            except Exception as e:
                send_status = f"邮件计划发送失败: {str(e)}"
                print(send_status)
    
    # 返回完整的报告内容和文件名
    return {
        "report_filename": report_filename,
        "email_status": send_status,
        "report_preview": report_content  # 返回完整报告内容，不做截断
    }

# 运行分析的简化函数（供app.py调用）
def run_news_analysis(keywords, email, time_range_days=30, send_time="now", ensure_chinese=False):
    """
    运行新能源汽车新闻分析（简化函数接口）
    """
    try:
        result = run_ev_news_analysis(
            keywords=keywords, 
            email=email, 
            time_range_days=time_range_days, 
            send_time=send_time,
            ensure_chinese=ensure_chinese
        )
        return result
    except Exception as e:
        error_message = f"分析过程中发生错误: {str(e)}"
        print(error_message)
        return {"error": error_message}

# 测试代码
if __name__ == "__main__":
    # 测试搜索和报告生成
    result = run_ev_news_analysis(
        keywords="比亚迪 电池技术",
        time_range_days=30
    )
    print(result["report_preview"]) 