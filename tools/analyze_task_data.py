# analyze_dataset.py
"""
Reads the generated task_request_data.csv and performs a detailed
statistical analysis to validate the characteristics of the dataset.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_FILENAME = "../data/task_request_data.csv"


def analyze_dataset():
    print(f"--- Analyzing Dataset: {INPUT_FILENAME} ---")
    try:
        df = pd.read_csv(INPUT_FILENAME)
    except FileNotFoundError:
        print(f"ERROR: Data file not found. Please run 'create_task_dataset.py' first.")
        return

    print(f"\n{'=' * 70}")
    print("DATASET STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total requests: {len(df):,}")

    # --- 1. Service Distribution ---
    print(f"\n📊 Service Distribution (top 10):")
    service_counts = df['service'].value_counts()
    for i, (service, count) in enumerate(service_counts.head(10).items()):
        pct = (count / len(df)) * 100
        print(f"  Service {service:2d}: {count:8,} ({pct:5.2f}%)")

    # --- 2. Temporal Locality ---
    print("\nCalculating temporal locality...")
    repeats = (df['service'].to_numpy()[:-1] == df['service'].to_numpy()[1:]).sum()
    actual_locality = repeats / (len(df) - 1)
    print(f"🔁 Temporal Locality (repeat rate): {actual_locality:.2%}")

    # --- 3. Zipf Skew ---
    most_common = service_counts.iloc[0]
    least_common = service_counts.iloc[-1]
    print(f"\n📈 Zipf Skew Ratio: {most_common / least_common:.2f}x (most/least common)")

    # --- 4. Service Transition Analysis ---
    # This was the slow part. We will optimize it.
    print(f"\n🔀 Analyzing Service Transitions (this may take a moment)...")
    df['next_service'] = df['service'].shift(-1)
    # Create pairs of (service, next_service) and count them
    transition_counts = df.groupby(['service', 'next_service']).size().reset_index(name='count')

    # Calculate probabilities for each transition
    service_totals = df['service'].value_counts().reset_index()
    service_totals.columns = ['service', 'total_count']
    transition_counts = pd.merge(transition_counts, service_totals, on='service')
    transition_counts['probability'] = transition_counts['count'] / transition_counts['total_count']

    strong_transitions = transition_counts[
        (transition_counts['probability'] > 0.3) &
        (transition_counts['service'] != transition_counts['next_service'])
        ].sort_values(by='probability', ascending=False)

    print(f"  Found {len(strong_transitions)} strong non-self transitions (>30% probability)")
    if not strong_transitions.empty:
        print(f"  Top 5 transitions:")
        for _, row in strong_transitions.head(5).iterrows():
            print(f"    Service {int(row['service'])} → {int(row['next_service'])}: {row['probability']:.1%}")

    # --- 5. Content-Service Correlation ---
    print(f"\n🔗 Analyzing Content-Service Correlation...")
    df_with_content = df.dropna(subset=['content'])
    if not df_with_content.empty:
        correlation_counts = df_with_content.groupby(['content', 'service']).size().reset_index(name='count')
        content_totals = df_with_content['content'].value_counts().reset_index()
        content_totals.columns = ['content', 'total_count']
        correlation_counts = pd.merge(correlation_counts, content_totals, on='content')
        correlation_counts['probability'] = correlation_counts['count'] / correlation_counts['total_count']

        # Find the most dominant service for each content type
        dominant_pairs = correlation_counts.loc[correlation_counts.groupby('content')['probability'].idxmax()]
        dominant_pairs = dominant_pairs.sort_values(by='probability', ascending=False)

        print(f"  Top 5 most correlated content-service pairs:")
        for _, row in dominant_pairs.head(5).iterrows():
            print(
                f"    Content {int(row['content']):2d} is requested with Service {int(row['service']):2d} ({row['probability']:.1%} of the time)")

    print(f"\n{'-' * 30} VISUALIZATIONS {'-' * 30}")

    # --- Visualizations ---
    # Service Distribution Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=service_counts.index, y=service_counts.values, palette='viridis')
    plt.title('Overall Service Popularity Distribution', fontsize=16)
    plt.xlabel('Service ID')
    plt.ylabel('Number of Requests')
    plt.savefig('analysis_service_distribution.png')
    print("  ✅ Saved service distribution plot to 'analysis_service_distribution.png'")
    plt.close()

    # Transition Heatmap for top services
    top_n = 10
    top_services = service_counts.head(top_n).index
    heatmap_data = transition_counts[
        (transition_counts['service'].isin(top_services)) &
        (transition_counts['next_service'].isin(top_services))
        ].pivot(index='service', columns='next_service', values='probability').fillna(0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap='rocket')
    plt.title(f'Transition Probability Matrix (Top {top_n} Services)', fontsize=16)
    plt.xlabel('Next Service')
    plt.ylabel('Current Service')
    plt.savefig('analysis_transition_heatmap.png')
    print("  ✅ Saved transition heatmap to 'analysis_transition_heatmap.png'")
    plt.close()

    print(f"\n{'-' * 70}\n")


if __name__ == "__main__":
    analyze_dataset()
