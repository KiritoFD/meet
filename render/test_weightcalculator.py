from render.weight import WeightCalculatorBenchmark, create_test_data
from render.weight.performance import PerformanceAnalyzer

def print_model_recommendations(recommendations):
    """打印模型规模建议"""
    print("\n=== 模型规模建议 ===")
    print(f"推荐规模: {recommendations['推荐模型规模'][0]}")
    print(f"最大规模: {recommendations['推荐模型规模'][1]}")
    print("\n性能说明:")
    print("- 小型模型 (1,000顶点): 适合实时预览和快速测试")
    print("- 中型模型 (10,000顶点): 适合一般场景和角色模型")
    print("- 大型模型 (50,000顶点): 适合复杂模型，可能需要较长处理时间")
    print("- 自定义规模: 根据具体需求自定义，注意性能警告")

def get_user_choice():
    """获取用户选择"""
    while True:
        print("\n请选择模型规模:")
        print("1. 小型模型 (1,000 顶点) - 适合快速预览")
        print("2. 中型模型 (10,000 顶点) - 适合一般使用")
        print("3. 大型模型 (50,000 顶点) - 适合复杂模型")
        print("4. 自定义规模")
        print("5. 退出程序")
        
        choice = input("\n请输入选择 (1-5): ")
        
        if choice == '5':
            return None, None
        elif choice == '4':
            try:
                vertex_count = int(input("请输入顶点数量: "))
                if vertex_count <= 0:
                    print("顶点数量必须大于0")
                    continue
                return vertex_count, "自定义"
            except ValueError:
                print("请输入有效的数字")
                continue
        elif choice in ['1', '2', '3']:
            sizes = {
                '1': (1000, "小型"),
                '2': (10000, "中型"),
                '3': (50000, "大型")
            }
            return sizes[choice]
        else:
            print("请输入有效的选项 (1-5)")

def main():
    """主函数"""
    # 创建性能分析器
    analyzer = PerformanceAnalyzer()
    
    # 分析系统性能并获取建议
    recommendations = analyzer.analyze_system()
    
    # 打印建议
    print_model_recommendations(recommendations)
    
    while True:
        # 获取用户选择
        result = get_user_choice()
        if result is None:
            print("\n程序已退出")
            break
            
        vertex_count, size_name = result
        
        # 获取配置建议
        config, risk_level = analyzer.suggest_config(vertex_count)
        
        print(f"\n选择的模型: {size_name}模型 ({vertex_count:,} 顶点)")
        print(f"风险等级: {risk_level}")
        
        if risk_level == 'high':
            proceed = input("警告：该规模可能影响性能，是否继续？(y/n): ")
            if proceed.lower() != 'y':
                print("已取消测试")
                continue
        
        # 运行测试
        benchmark = WeightCalculatorBenchmark()
        vertices, joints = create_test_data(
            vertex_count, 
            min(30, vertex_count // 1000 + 5)
        )
        results = benchmark.run_benchmark(vertices, joints, config)
        
        # 询问是否继续测试
        if input("\n是否继续测试其他规模？(y/n): ").lower() != 'y':
            print("\n程序已退出")
            break

if __name__ == "__main__":
    main() 