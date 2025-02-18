from pose.benchmark import WeightCalculatorBenchmark, create_test_data

# 创建小型测试数据
vertex_count = 1000  # 1000个顶点
joint_count = 10     # 10个骨骼
vertices, joints = create_test_data(vertex_count, joint_count)

# 运行测试
benchmark = WeightCalculatorBenchmark()
results = benchmark.run_benchmark(vertices, joints)
