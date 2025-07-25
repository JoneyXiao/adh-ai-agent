import os
from bs4 import BeautifulSoup
from multiprocessing import Process, Queue
from patchright.sync_api import sync_playwright


from mcp.server.fastmcp import FastMCP


# 初始化 MCP 服务器
mcp = FastMCP("WeatherServer", host="127.0.0.1", port=8003)


def get_page_content(url: str, q: Queue):
    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch(
            headless=True,
            args=['--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36']
        )
        page = browser.new_page()
        page.goto(url, timeout=30000)
        page_content = page.content()
        browser.close()
        q.put(page_content)


def extract_1_week_weather(page_content):
    soup = BeautifulSoup(page_content, "html.parser")

        # Remove scripts and styles
    for tag in soup(['script', 'style']):
        tag.decompose()

    weather_list = []

    for item in soup.find_all("div", class_="weatherWrap"):
        date_and_week = item.find("div", class_="date").get_text().strip() # type: ignore
        desc_list = item.find_all("div", class_="desc") # type: ignore
        forenoon_desc = desc_list[0].get_text().strip()
        afternoon_desc = desc_list[1].get_text().strip()
        temp_list = item.find_all("div", class_="tmp") # type: ignore
        temp_high = temp_list[0].get_text().strip()
        temp_low = temp_list[1].get_text().strip()
        wind_desc = item.find("div", class_="winds").get_text().strip() # type: ignore

        weather = {
                "date": date_and_week,
                "forenoon": forenoon_desc,
                "afternoon": afternoon_desc,
                "temp_high": temp_high,
                "temp_low": temp_low,
                "wind_speed": wind_desc
            }
            
        weather_list.append(weather)

    return weather_list
    

def get_jzg_1_week_weather() -> list:
    url = "https://www.nmc.cn/publish/forecast/ASC/jiuzhaigou.html"

    try:
        q = Queue()
        p = Process(target=get_page_content, args=(url,q))
        p.start()
        # p.join()
        page_content = q.get()
        
        weather_list = extract_1_week_weather(page_content)
        return weather_list
    except Exception as e:
        return []


def get_weather(city_name: str) -> str:
    """根据城市中文名返回当前天气中文描述"""

    if city_name.lower() != "jiuzhaigou":
        return "查询出错：此城市 {city_name} 暂时不支持！"

    try:
        result = get_jzg_1_week_weather()
        weather_tomorrow = result[1]
        print(weather_tomorrow)

        return (
            f"城市: Jiuzhaigou\n"
            f"上午天气: {weather_tomorrow['forenoon']}\n"
            f"下午天气: {weather_tomorrow['afternoon']}\n"
            f"最高温度: {weather_tomorrow['temp_high']}°C\n"
            f"最低温度: {weather_tomorrow['temp_low']}°C\n"
            f"风速: {weather_tomorrow['wind_speed']} m/s\n"
        )
    except Exception as e:
        return f"查询出错：{str(e)}"

# expose this tool
@mcp.tool('query_weather', '查询城市天气')
def query_weather(city: str) -> str:
    """
        输入指定城市的中文名称，返回当前天气查询结果。
        :param city: 城市名称
        :return: 格式化后的天气信息
        """
    return get_weather(city)


if __name__ == "__main__":
    # 以 SSE 方式运行 MCP 服务器
    mcp.run(transport='sse')
