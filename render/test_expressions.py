import time
import matplotlib.pyplot as plt
import numpy as np
import os

def test_realistic_expressions():
    """测试写实风格表情"""
    print("\n=== 写实表情测试开始 ===")
    
    controller = AnimationController()
    
    # 记录所有参数
    params = [
        'eye_open', 'eye_squint', 'eye_wide', 'eye_lower_lid',
        'brow_raise', 'brow_furrow', 'brow_tilt',
        'mouth_open', 'mouth_smile', 'mouth_frown', 'mouth_pucker',
        'cheek_raise', 'cheek_puff'
    ]
    
    times = []
    values = {param: [] for param in params}
    
    # 设置初始状态
    controller.set_expression('neutral')
    
    start_time = time.time()
    current_time = start_time
    frame_count = 0
    
    print("开始记录表情变化...")
    
    # 写实表情序列
    expressions = [
        (1.0, 'subtle_smile'),    # 淡淡微笑
        (2.0, 'surprised'),       # 惊讶
        (3.0, 'thinking'),        # 思考
        (4.0, 'warm_smile'),      # 温暖笑容
        (5.0, 'concerned'),       # 担忧
        (6.0, 'amused'),          # 被逗乐
        (7.0, 'skeptical'),       # 怀疑
        (8.0, 'neutral')          # 回到自然状态
    ]
    
    while current_time - start_time < 9.0:
        elapsed = current_time - start_time
        frame_count += 1
        
        # 检查是否需要切换表情
        for time_point, expression in expressions:
            if time_point <= elapsed < time_point + 0.02:
                print(f"{time_point}秒: 切换到{expression}表情")
                controller.set_expression(expression)
        
        # 更新并记录状态
        current_values = controller.update()
        times.append(elapsed)
        for param in params:
            values[param].append(current_values.get(param, 0.0))
        
        time.sleep(0.016)
        current_time = time.time()
    
    print(f"记录完成，共 {frame_count} 帧")
    
    # 绘制图表
    plt.figure(figsize=(15, 10))
    
    # 使用柔和的颜色方案
    colors = plt.cm.Spectral(np.linspace(0, 1, len(params)))
    for param, color in zip(params, colors):
        plt.plot(times, values[param], label=param, color=color, linewidth=1.5, alpha=0.7)
    
    # 添加表情切换时间线和标签
    for time_point, expression in expressions:
        plt.axvline(x=time_point, color='gray', linestyle='--', alpha=0.2)
        plt.text(time_point, 1.15, expression, rotation=45, ha='right')
    
    plt.xlabel('时间 (秒)')
    plt.ylabel('参数值')
    plt.title('写实风格面部表情参数变化')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.ylim(-0.1, 1.2)
    
    # 保存高质量图表
    output_path = 'realistic_expression_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {os.path.abspath(output_path)}")
    plt.close()
    
    print("\n=== 写实表情测试完成 ===")

if __name__ == "__main__":
    print("=== 开始测试 ===")
    try:
        test_realistic_expressions()
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
    print("=== 测试结束 ===") 