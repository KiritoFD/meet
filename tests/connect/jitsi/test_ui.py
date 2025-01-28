import pytest
from playwright.async_api import async_playwright

@pytest.mark.ui
@pytest.mark.asyncio
async def test_meeting_ui_flow():
    """测试完整的UI会议流程"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # 进入会议室页面
        await page.goto("http://localhost:3000/meeting/test-room")
        
        # 验证基础元素
        await page.wait_for_selector("#meeting-container")
        assert await page.is_visible("text=当前会议室：test-room")
        
        # 测试加入会议
        await page.click("#join-button")
        await page.wait_for_selector("#participant-list")
        
        # 测试发送消息
        await page.fill("#message-input", "Hello World")
        await page.click("#send-button")
        await page.wait_for_selector("text=Hello World")
        
        # 测试离开会议
        await page.click("#leave-button")
        await page.wait_for_selector("text=已离开会议室")
        
        await browser.close() 