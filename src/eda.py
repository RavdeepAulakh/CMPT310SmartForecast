import matplotlib.pyplot as plt
import seaborn as sns


def plot_score_distribution(df):
    plt.hist(df["exam_score"], bins=20)
    plt.title("Distribution of Exam Scores")
    plt.xlabel("Exam Score")
    plt.ylabel("Count")
    plt.show()


def plot_correlations(df):
    corr = df.drop(columns=["student_id"]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def scatter_relationships(df):
    features = ["hours_studied", "sleep_hours",
                "attendance_percent", "previous_scores"]

    for feature in features:
        plt.scatter(df[feature], df["exam_score"])
        plt.xlabel(feature)
        plt.ylabel("exam_score")
        plt.title(f"{feature} vs Exam Score")
        plt.show()