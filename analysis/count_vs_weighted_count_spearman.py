import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# -----------------------------------------------------------------------------
# Example function to create plot for a given patient, eye, and region
# -----------------------------------------------------------------------------
def create_count_vs_intensity_plot(
    df: pd.DataFrame,
    patient: str,
    eye: str,
    region: str,
    output_folder: str = "Count for aggregate severity diagnosis"
):
    """
    For the specified patient, eye, and region, compute:
      1. number of row entries for each date;
      2. normalized number of positive row entries;
      3. weighted normalized number of positive row entries;
    Then produce a scatter plot (y-axis=normalized positives, x-axis=weighted normalized),
    compute Spearman's correlation, display it on the plot, and save to disk.
    """

    # Filter the DataFrame to the desired subset
    mask = (
        (df["patient"] == patient) &
        (df["eye"] == eye) &
        (df["region"] == region)
    )
    df_subset = df.loc[mask].copy()

    if df_subset.empty:
        print(f"No rows match {patient}, {eye}, {region}")
        return

    # Define severity mapping
    severity_scores = {
        "negative": 0,
        "mild": 1,
        "moderate": 2,
        "severe": 3
    }

    # Prepare grouped stats
    grouped = df_subset.groupby("date")

    results = []
    for date_val, group_data in grouped:
        # (1) total number of rows
        total_rows = len(group_data)

        # (2) normalized number of positive rows
        #     positives = rows whose Label ∈ {mild, moderate, severe}
        positives = group_data["Label"].isin(["mild", "moderate", "severe"]).sum()
        norm_positive = positives / total_rows

        # (3) weighted normalized number of positive rows
        #     sum(s_category * count_category) / (3 * total_rows)
        #     where s_category = 0,1,2,3 for negative/mild/moderate/severe
        score_sum = 0
        for label_cat, score_val in severity_scores.items():
            count_cat = (group_data["Label"] == label_cat).sum()
            score_sum += score_val * count_cat
        weighted_norm = score_sum / (3 * total_rows)

        results.append({
            "date": date_val,
            "total_rows": total_rows,
            "normalized_positive": norm_positive,
            "weighted_normalized": weighted_norm
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print(f"No results computed for {patient}, {eye}, {region}")
        return

    # -------------------------------------------------------------------------
    # Scatter plot
    # -------------------------------------------------------------------------
    x = results_df["weighted_normalized"]
    y = results_df["normalized_positive"]

    # Compute Spearman's correlation
    rho, pval = spearmanr(x, y)

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    # plt.grid()
    plt.xlabel("Severity-Weighted Normalized Positives")
    plt.ylabel("Normalized Number of Positives")
    plt.title(f"{patient} – {eye} – {region} – by date")

    # Place correlation text on the plot
    plt.text(
        0.05, 0.95,
        f"Spearman r = {rho:.3f}\n(p={pval:.3g})",
        transform=plt.gca().transAxes,
        va="top",
        ha="left"
    )

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(
        output_folder,
        f"count_vs_severity_weighted_count_{patient}_{eye}_{region}.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the CSV file
    csv_file = "Annotation 01032025.csv"   # <-- adjust path as needed
    df = pd.read_csv(csv_file)

    # Parse out patient, date, eye, region from the Image File column
    # Example path structure: "Patient001/20210420/OD/AR/Patient001_20210420_OD_AR_0000.png"
    # That means: patient = parts[0], date = parts[1], eye = parts[2], region = parts[3]
    df["parts"] = df["Image File"].str.split('/')
    df["patient"] = df["parts"].apply(lambda x: x[0])
    df["date"] = df["parts"].apply(lambda x: x[1])
    df["eye"] = df["parts"].apply(lambda x: x[2])
    df["region"] = df["parts"].apply(lambda x: x[3])

    # Example call for a single combination: (Patient001, OD, AR)

    create_count_vs_intensity_plot(df, "Patient001", "OD", "AR")
    # create_count_vs_intensity_plot(df, "Patient001", "OD", "FO")
    # create_count_vs_intensity_plot(df, "Patient001", "OD", "ON")
    # create_count_vs_intensity_plot(df, "Patient001", "OD", "SA")
    # create_count_vs_intensity_plot(df, "Patient001", "OD", "IA")
    # create_count_vs_intensity_plot(df, "Patient001", "OS", "AR")
    # create_count_vs_intensity_plot(df, "Patient001", "OS", "FO")
    # create_count_vs_intensity_plot(df, "Patient001", "OS", "ON")
    # create_count_vs_intensity_plot(df, "Patient001", "OS", "SA")
    # create_count_vs_intensity_plot(df, "Patient001", "OS", "IA")
