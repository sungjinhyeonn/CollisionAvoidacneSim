import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns





def compare_metrics_by_scenario():
    # 결과를 저장할 데이터프레임 초기화
    all_results = pd.DataFrame()
    
    # 모든 시나리오 데이터 수집
    scenario_dirs = sorted(glob.glob('log/scenario_*'))
    
    for scenario_dir in scenario_dirs:
        csv_path = os.path.join(scenario_dir, 'all_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_results = pd.concat([all_results, df])

    # 시나리오별 상세 비교
    metrics = ['collision_count', 'execution_time', 'operation_time', 'total_distance']
    
    # 각 메트릭별로 별도의 그래프 생성
    for metric in metrics:
        plt.figure(figsize=(15, 8))  # 그래프 크기 증가
        
        # 시나리오별 알고리즘 비교
        pivot_data = all_results.pivot(index='scenario', columns='algorithm', values=metric)
        
        # 막대 그래프 생성
        ax = pivot_data.plot(kind='bar', width=0.6)  # 막대 폭 감소
        
        plt.title(f'{metric.replace("_", " ").title()} Comparison by Scenario', fontsize=14, pad=20)
        plt.xlabel('Scenario Number', fontsize=12, labelpad=10)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12, labelpad=10)
        
        # x축 레이블 간격 조정
        plt.xticks(rotation=0)  # 회전 제거
        
        # 범례 위치 및 스타일 조정
        plt.legend(title='Algorithm', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        
        # 그리드 추가
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 여백 조정
        plt.tight_layout()
        
        # PNG로 저장
        plt.savefig(f'comparison_{metric}.png', bbox_inches='tight', dpi=300)
        plt.close()

    # 모든 메트릭을 하나의 그래프로 통합
    plt.figure(figsize=(20, 15))
    
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, idx)
        pivot_data = all_results.pivot(index='scenario', columns='algorithm', values=metric)
        pivot_data.plot(kind='bar', ax=plt.gca(), width=0.6)  # 막대 폭 감소
        
        plt.title(f'{metric.replace("_", " ").title()}', fontsize=14, pad=20)
        plt.xlabel('Scenario Number', fontsize=12, labelpad=10)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12, labelpad=10)
        
        # x축 레이블 간격 조정
        plt.xticks(rotation=0)  # 회전 제거
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Algorithm', fontsize=10)
    
    # 전체 그래프 여백 조정
    plt.tight_layout(pad=3.0)  # 여백 증가
    plt.savefig('all_metrics_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 알고리즘별 평균 성능 비교 그래프 추가
    def create_algorithm_comparison():
        metrics = ['execution_time', 'operation_time', 'total_distance']
        
        # 알고리즘별 평균 계산
        avg_performance = all_results.groupby('algorithm')[metrics].mean()
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        
        # 막대 그래프 생성
        bar_width = 0.25
        index = range(len(avg_performance.index))
        
        for i, metric in enumerate(metrics):
            plt.bar([x + i*bar_width for x in index], 
                   avg_performance[metric], 
                   bar_width, 
                   label=metric.replace('_', ' ').title())
        
        # 그래프 스타일링
        plt.title('Average Performance by Algorithm', fontsize=14, pad=20)
        plt.xlabel('Algorithm', fontsize=12, labelpad=10)
        plt.ylabel('Average Value', fontsize=12, labelpad=10)
        
        # x축 레이블 설정
        plt.xticks([x + bar_width for x in index], avg_performance.index)
        
        # 범례 추가
        plt.legend()
        
        # 그리드 추가
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 여백 조정
        plt.tight_layout()
        
        # 값 레이블 추가
        for i, metric in enumerate(metrics):
            for j, value in enumerate(avg_performance[metric]):
                plt.text(j + i*bar_width, value, f'{value:.1f}', 
                        ha='center', va='bottom')
        
        # PNG로 저장
        plt.savefig('algorithm_average_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()

        # 수치 데이터 출력
        print("\nAlgorithm Average Performance:")
        print(avg_performance.round(2))
        
        # CSV로도 저장
        avg_performance.round(2).to_csv('algorithm_average_performance.csv')

    # 기존의 그래프들 생성 후 평균 비교 그래프 생성
    create_algorithm_comparison()

# 두 그래프 모두 생성


if __name__ == "__main__":
    compare_metrics_by_scenario()
