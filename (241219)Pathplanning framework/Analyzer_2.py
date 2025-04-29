import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

def create_analysis_folder():
    """분석 결과를 저장할 폴더 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = f'analysis_results_{timestamp}'
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def plot_paper_style_graphs(df, output_folder):
    """논문 스타일의 그래프들 생성"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
    plt.style.use('seaborn-whitegrid')
    
    # 알고리즘 순서와 색상 고정
    algorithms = ['SAC_DTP', 'SAC']
    colors = ['black', 'blue', 'red', 'green', 'purple']
    x = np.arange(len(algorithms))
    
    # 1. Navigation Time Plot
    for i, alg in enumerate(algorithms):
        data = df[df['algorithm'] == alg]['operation_time']
        ax1.errorbar(i, data.mean(), yerr=data.std(), fmt='o', color=colors[i], 
                    capsize=5, capthick=1, label=alg)
    means = [df[df['algorithm'] == alg]['operation_time'].mean() for alg in algorithms]
    ax1.plot(x, means, color='gray', linestyle='-', alpha=0.5)
    ax1.set_ylim(0, 200)  # y축 범위 설정
    
    # 2. Path Length Plot
    for i, alg in enumerate(algorithms):
        data = df[df['algorithm'] == alg]['total_distance']
        ax2.errorbar(i, data.mean(), yerr=data.std(), fmt='o', color=colors[i], 
                    capsize=5, capthick=1, label=alg)
    means = [df[df['algorithm'] == alg]['total_distance'].mean() for alg in algorithms]
    ax2.plot(x, means, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylim(0, 200)  # y축 범위 설정
    
    # 3. Collision Count Plot
    for i, alg in enumerate(algorithms):
        data = df[df['algorithm'] == alg]['collision_count']
        # 모든 데이터 포인트를 scatter plot으로 표시
        ax3.scatter([i] * len(data), data, color=colors[i], alpha=0.5)
        # 평균값 표시
        ax3.plot(i, data.mean(), 'o', color=colors[i], markersize=10)
    
    # 평균값들을 선으로 연결
    means = [df[df['algorithm'] == alg]['collision_count'].mean() for alg in algorithms]
    ax3.plot(x, means, color='gray', linestyle='-', alpha=0.5)
    ax3.set_ylim(0, 10)  # y축 범위를 0부터 10으로 설정
    
    # 4. Performance Score
    scores = {}
    for alg in algorithms:
        means = df[df['algorithm'] == alg].agg({
            'operation_time': 'mean',
            'total_distance': 'mean',
            'collision_count': 'mean'
        })
        
        # 가중합의 역수로 점수 계산
        weighted_sum = (5 * means['operation_time'] + 
                       0.2 * means['total_distance'] + 
                       8 * means['collision_count'])
        scores[alg] = 1 / weighted_sum
    
    # 성능 점수 그래프 그리기
    bars = ax4.bar(x, [scores[alg] for alg in algorithms])
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])
        bar.set_alpha(0.6)
    
    # 그래프 설정
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45)
        ax.grid(True)
    
    ax1.set_ylabel('Navigation Time (s)')
    ax2.set_ylabel('Path Length (m)')
    ax3.set_ylabel('Collision Count')
    ax4.set_ylabel('Performance Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms, rotation=45)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_scenarios(output_folder):
    """모든 시나리오 분석"""
    # 모든 시나리오 폴더 찾기
    scenario_dirs = sorted([d for d in os.listdir('log') if d.startswith('scenario_')])
    
    # 각 시나리오별 분석
    for scenario_dir in scenario_dirs:
        print(f"\n\nAnalyzing {scenario_dir}")
        print("="*80)
        
        scenario_results = f'./log/{scenario_dir}/all_results.csv'
        if os.path.exists(scenario_results):
            # 데이터 로드 및 전처리
            df = pd.read_csv(scenario_results)
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # 시나리오별 서브폴더 생성
            scenario_folder = os.path.join(output_folder, scenario_dir)
            os.makedirs(scenario_folder, exist_ok=True)
            
            # 논문 스타일 그래프 생성
            plot_paper_style_graphs(df, scenario_folder)
            
            # 통계 분석
            metrics = ['total_distance', 'collision_count', 'min_obstacle_distance', 
                      'operation_time', 'execution_time']
            
            stats = df.groupby('algorithm')[metrics].agg(['mean', 'std', 'min', 'max']).round(2)
            
            # 결과 출력
            print("\nStatistical Analysis by Algorithm:")
            print("="*80)
            for algorithm in df['algorithm'].unique():
                print(f"\n{algorithm}:")
                print("-"*40)
                alg_stats = stats.loc[algorithm]
                for metric in metrics:
                    mean = alg_stats[metric]['mean']
                    std = alg_stats[metric]['std']
                    print(f"{metric}:")
                    print(f"  Mean ± Std: {mean:.2f} ± {std:.2f}")
            
            # 결과 파일 저장
            df.to_csv(os.path.join(scenario_folder, 'raw_data.csv'), index=False)
            stats.to_csv(os.path.join(scenario_folder, 'statistics.csv'))

def analyze_overall_results(output_folder):
    """모든 시나리오의 전체 평균 분석"""
    print("\n\nAnalyzing Overall Results (All Scenarios)")
    print("="*80)
    
    # 모든 시나리오의 결과 데이터 수집
    all_data = []
    scenario_dirs = sorted([d for d in os.listdir('log') if d.startswith('scenario_')])
    
    for scenario_dir in scenario_dirs:
        scenario_results = f'./log/{scenario_dir}/all_results.csv'
        if os.path.exists(scenario_results):
            df = pd.read_csv(scenario_results)
            df['scenario'] = scenario_dir  # 시나리오 정보 추가
            all_data.append(df)
    
    if all_data:
        # 모든 데이터 합치기
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        
        # 전체 평균 분석 폴더 생성
        overall_folder = os.path.join(output_folder, 'overall_analysis')
        os.makedirs(overall_folder, exist_ok=True)
        
        # 논문 스타일 그래프 생성 (전체 평균)
        plot_paper_style_graphs(combined_df, overall_folder)
        
        # 통계 분석
        metrics = ['total_distance', 'collision_count', 'min_obstacle_distance', 
                  'operation_time', 'execution_time']
        
        # 알고리즘별 전체 통계
        overall_stats = combined_df.groupby('algorithm')[metrics].agg(['mean', 'std', 'min', 'max']).round(2)
        
        # 시나리오별, 알고리즘별 평균 계산
        scenario_means = combined_df.groupby(['scenario', 'algorithm'])[metrics].mean()
        algorithm_means = scenario_means.groupby('algorithm').agg(['mean', 'std'])
        
        # 결과 출력
        print("\nOverall Statistical Analysis by Algorithm (Across All Scenarios):")
        print("="*80)
        for algorithm in combined_df['algorithm'].unique():
            print(f"\n{algorithm}:")
            print("-"*40)
            alg_stats = algorithm_means.loc[algorithm]
            for metric in metrics:
                mean = alg_stats[metric]['mean']
                std = alg_stats[metric]['std']
                print(f"{metric}:")
                print(f"  Mean ± Std: {mean:.2f} ± {std:.2f}")
        
        # 결과 파일 저장
        combined_df.to_csv(os.path.join(overall_folder, 'all_scenarios_data.csv'), index=False)
        overall_stats.to_csv(os.path.join(overall_folder, 'overall_statistics.csv'))
        algorithm_means.to_csv(os.path.join(overall_folder, 'algorithm_means_by_scenario.csv'))

def analyze_results():
    """결과 분석 및 저장"""
    # 분석 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f'analysis_results_{timestamp}'
    os.makedirs(output_folder, exist_ok=True)
    
    # 개별 시나리오 분석
    analyze_all_scenarios(output_folder)
    
    # 전체 결과 분석
    analyze_overall_results(output_folder)
    
    print(f"\nAnalysis completed! Results saved in: {output_folder}")
    return output_folder

if __name__ == "__main__":
    output_folder = analyze_results()