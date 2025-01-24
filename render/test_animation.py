import time
import matplotlib.pyplot as plt
import numpy as np
import os
from animation.controller import AnimationController

def test_expressions():
    """测试多种表情"""
    print("\n=== 表情测试开始 ===")
    
    controller = AnimationController()
    
    # 记录所有参数
    params = [
        'eye_open', 'eye_happy', 'eye_surprised',
        'brow_raise', 'brow_angry',
        'mouth_open', 'mouth_smile', 'mouth_pout'
    ]
    
    times = []
    values = {param: [] for param in params}
    
    # 设置初始状态
    controller.set_expression('neutral')
    
    start_time = time.time()
    current_time = start_time
    frame_count = 0
    
    print("开始记录表情变化...")
    
    # 表情序列
    expressions = [
        (1.0, 'surprised'),
        (2.0, 'happy'),
        (3.0, 'angry'),
        (4.0, 'sad'),
        (5.0, 'neutral')
    ]
    
    while current_time - start_time < 6.0:
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
    plt.figure(figsize=(15, 8))
    
    # 使用不同颜色绘制各个参数
    colors = plt.cm.rainbow(np.linspace(0, 1, len(params)))
    for param, color in zip(params, colors):
        plt.plot(times, values[param], label=param, color=color, linewidth=1.5)
    
    # 添加表情切换时间线
    for time_point, expression in expressions:
        plt.axvline(x=time_point, color='gray', linestyle='--', alpha=0.3)
        plt.text(time_point, 1.1, expression, rotation=45)
    
    plt.xlabel('时间 (秒)')
    plt.ylabel('参数值')
    plt.title('面部表情参数变化')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 1.2)
    
    # 保存图表
    output_path = 'expression_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {os.path.abspath(output_path)}")
    plt.close()
    
    print("\n=== 表情测试完成 ===")

if __name__ == "__main__":
    print("=== 开始测试 ===")
    try:
        test_expressions()  # 使用新的测试函数
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
    print("=== 测试结束 ===") 