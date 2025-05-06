import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(df):
    # Year-wise disaster count
    df["YEAR"] = df["EVENT START DATE"].dt.year
    sns.countplot(data=df, x="YEAR")
    plt.xticks(rotation=90)
    plt.title("Number of Disasters Per Year")
    plt.tight_layout()
    plt.show()

    # Distribution of Event Duration
    sns.histplot(df["EVENT DURATION"], bins=30, kde=True)
    plt.title("Distribution of Event Duration")
    plt.tight_layout()
    plt.show()

    # Top 10 Costliest Disasters
    top_costly = df.sort_values("ESTIMATED TOTAL COST", ascending=False).head(10)
    sns.barplot(data=top_costly, x="PLACE", y="ESTIMATED TOTAL COST")
    plt.title("Top 10 Costliest Disasters by Location")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Correlation Matrix
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.show()
