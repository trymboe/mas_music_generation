import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from scipy.stats import ttest_ind
from scipy.stats import shapiro
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

SONG_MAPPING = {
    1: ["bad", "high", "1"],
    2: ["good", "high", "1"],
    3: ["bad", "low", "3"],
    4: ["good", "mid", "2"],
    5: ["bad", "mid", "2"],
    6: ["good", "high", "6"],
    7: ["good", "mid", "3"],
    8: ["bad", "low", "1"],
    9: ["good", "low", "1"],
    10: ["bad", "high", "4"],
    11: ["good", "high", "2"],
    12: ["bad", "mid", "3"],
    13: ["good", "mid", "4"],
    14: ["bad", "mid", "4"],
    15: ["bad", "low", "4"],
    16: ["bad", "high", "2"],
    17: ["bad", "low", "2"],
    18: ["bad", "mid", "1"],
    19: ["bad", "mid", "1"],
    20: ["good", "mid", "1"],
    21: ["good", "low", "4"],
    22: ["good", "low", "2"],
    23: ["good", "low", "3"],
    24: ["bad", "high", "3"],
    25: ["good", "high", "3"],
}

answers = []

good = []
bad = []

good_high = []
good_mid = []
good_low = []

bad_high = []
bad_mid = []
bad_low = []

high = []
mid = []
low = []

pro = []
non_pro = []

male = []
female = []

interested = []
non_interested = []

interested_non_pro = []


def get_data():
    df = pd.read_csv("data/responses.csv")
    for idx, row in df.iterrows():
        elements = row[3:7].tolist()
        answers.append(elements)
    df = df.iloc[:, 7:]

    for idx, start_col in enumerate(range(0, len(df.columns), 4)):
        if idx == 18:
            continue
        end_col = start_col + 4
        df_subset = df.iloc[:, start_col:end_col]
        extract_first_number = lambda x: int(re.search(r"\d+", x).group())

        result_lists = []
        for index, row in df_subset.iterrows():

            # Extract and convert numeric values
            numeric_values = [extract_first_number(cell) for cell in row]
            # Calculate average
            avg_value = sum(numeric_values) / len(numeric_values)
            # Append average to the list of values
            row_list = numeric_values + [avg_value]

            if int(answers[index][3][0]) > 2:
                pro.append(row_list)
            else:
                non_pro.append(row_list)

            if int(answers[index][1][0]) == 1:
                male.append(row_list)
            else:
                female.append(row_list)

            if int(answers[index][2][0]) > 3:
                interested.append(row_list)
            else:
                non_interested.append(row_list)

            if int(answers[index][2][0]) > 3 and int(answers[index][3][0]) < 3:
                interested_non_pro.append(row_list)

            if SONG_MAPPING[idx + 1][0] == "good":
                good.append(row_list)
                if SONG_MAPPING[idx + 1][1] == "high":
                    good_high.append(row_list)
                    high.append(row_list)
                elif SONG_MAPPING[idx + 1][1] == "mid":
                    good_mid.append(row_list)
                    mid.append(row_list)
                else:
                    good_low.append(row_list)
                    low.append(row_list)
            else:
                bad.append(row_list)
                if SONG_MAPPING[idx + 1][1] == "high":
                    bad_high.append(row_list)
                    high.append(row_list)
                elif SONG_MAPPING[idx + 1][1] == "mid":
                    bad_mid.append(row_list)
                    mid.append(row_list)
                else:
                    bad_low.append(row_list)
                    low.append(row_list)


def get_results(printers, print_pro, print_interest, print_gender, print_age):
    strings = [
        "Harmony scores",
        "Rhythm scores",
        "Musical intention scores",
        "Subjective evaluation scores",
        "Average scores",
    ]
    for idx, text in enumerate(strings):
        if printers[idx]:
            print(f"--------------{text}--------------")
            row_averages = [row[idx] for row in good]
            overall_average = sum(row_averages) / len(row_averages)
            print("good average score:", overall_average)

            row_averages = [row[idx] for row in bad]
            overall_average = sum(row_averages) / len(row_averages)
            print("bad average score:", overall_average)
            print()

            row_averages = [row[idx] for row in high]
            overall_average = sum(row_averages) / len(row_averages)
            print("high average score:", overall_average)

            row_averages = [row[idx] for row in mid]
            overall_average = sum(row_averages) / len(row_averages)
            print("mid average score:", overall_average)

            row_averages = [row[idx] for row in low]
            overall_average = sum(row_averages) / len(row_averages)
            print("low average score:", overall_average)
            print()

            row_averages = [row[idx] for row in good_high]
            overall_average = sum(row_averages) / len(row_averages)
            print("good_high average score:", overall_average)

            row_averages = [row[idx] for row in bad_high]
            overall_average = sum(row_averages) / len(row_averages)
            print("bad_high average score:", overall_average)
            print()

            row_averages = [row[idx] for row in good_mid]
            overall_average = sum(row_averages) / len(row_averages)
            print("good_mid average score:", overall_average)

            row_averages = [row[idx] for row in bad_mid]
            overall_average = sum(row_averages) / len(row_averages)
            print("bad_mid average score:", overall_average)
            print()

            row_averages = [row[idx] for row in good_low]
            overall_average = sum(row_averages) / len(row_averages)
            print("good_low average score:", overall_average)

            row_averages = [row[idx] for row in bad_low]
            overall_average = sum(row_averages) / len(row_averages)

            print("bad_low average score:", overall_average)
            print()
            print()

    if print_pro:
        print("--------------Pro--------------")
        for idx, text in enumerate(strings):
            row_averages = [row[idx] for row in pro]
            overall_average = sum(row_averages) / len(row_averages)
            print(f"Pro {text} average score:", overall_average)
        print()
        for idx, text in enumerate(strings):
            row_averages = [row[idx] for row in non_pro]
            overall_average = sum(row_averages) / len(row_averages)
            print(f"Non_pro {text} average score:", overall_average)
        print()

    if print_interest:
        print("--------------Interest--------------")
        for idx, text in enumerate(strings):
            row_averages = [row[idx] for row in interested]
            overall_average = sum(row_averages) / len(row_averages)
            print(f"Interested {text} average score:", overall_average)
        print()
        for idx, text in enumerate(strings):
            row_averages = [row[idx] for row in non_interested]
            overall_average = sum(row_averages) / len(row_averages)
            print(f"Not interested {text} average score:", overall_average)
        print()

    if print_gender:
        print("--------------Gender--------------")
        for idx, text in enumerate(strings):
            row_averages = [row[idx] for row in male]
            overall_average = sum(row_averages) / len(row_averages)
            print(f"Male {text} average score:", overall_average)
        print()
        for idx, text in enumerate(strings):
            row_averages = [row[idx] for row in female]
            overall_average = sum(row_averages) / len(row_averages)
            print(f"Female {text} average score:", overall_average)
        print()

    print("--------------Interested_non_pro--------------")
    for idx, text in enumerate(strings):
        row_averages = [row[idx] for row in interested_non_pro]
        overall_average = sum(row_averages) / len(row_averages)
        print(f"Interested non pro {text} average score:", overall_average)
    print()


def plot_figures(good, bad, title):
    titles = [
        "Harmony",
        "Rhythm",
        "Musical Intention",
        "Subjective Evaluation",
        "Average",
    ]
    p_values_df = statistical_test(good, bad, "w")
    for i in range(len(p_values_df)):
        if p_values_df.iloc[i]["P-Value"] < 0.05 / 15:
            titles[i] = titles[i] + "**"
        elif p_values_df.iloc[i]["P-Value"] < 0.05 / 5:
            titles[i] = titles[i] + "*"

    print(p_values_df)
    good_df = pd.DataFrame(
        good,
        columns=titles,
    )
    good_df["Type"] = "Unscrambled Coms"

    bad_df = pd.DataFrame(bad, columns=titles)
    bad_df["Type"] = "Scrambled Coms"

    # Concatenating both DataFrames
    data = pd.concat([good_df, bad_df])

    # Melting the DataFrame to use with seaborn
    melted_data = pd.melt(
        data, id_vars=["Type"], var_name="Metric", value_name="Rating"
    )

    # Creating the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Metric",
        y="Rating",
        hue="Type",
        data=melted_data,
        split=True,
        inner="quartile",
    )
    # Set the title with a larger font size
    plt.title(title, fontsize=20)

    # Optionally set the font size for labels and ticks
    plt.xlabel("Metric", fontsize=18)
    plt.ylabel("Rating", fontsize=18)
    plt.xticks(fontsize=14, rotation=10)
    plt.yticks(fontsize=14)
    plt.title(title)
    plt.subplots_adjust(bottom=0.2, top=0.9)
    # plt.savefig(f"figures/{title}.png")
    plt.show()


def statistical_test(group1, group2, type):
    # Convert to DataFrame for easier handling
    good_df = pd.DataFrame(
        group1,
        columns=[
            "Harmony",
            "Rhythm",
            "Musical Intention",
            "Subjective Evaluation",
            "Average",
        ],
    )
    bad_df = pd.DataFrame(
        group2,
        columns=[
            "Harmony",
            "Rhythm",
            "Musical Intention",
            "Subjective Evaluation",
            "Average",
        ],
    )

    # Initialize a list for storing test results
    results = []

    # Loop through each column in the DataFrame
    for column in good_df.columns:
        # Perform t-test
        if type == "t":
            stat, p_value = ttest_ind(good_df[column], bad_df[column])
        if type == "w":
            stat, p_value = wilcoxon(good_df[column], bad_df[column])

        results.append((column, stat, p_value))

    # Display the results
    results_df = pd.DataFrame(
        results, columns=["Category", "Test-Statistic", "P-Value"]
    )
    return results_df


def shapiro_wilk_test(data):
    for i in range(len(data)):

        statistic, p_value = shapiro(data[i])
        print("Shapiro-Wilk Test Statistic:", statistic)
        print("P-value:", p_value)

        if p_value > 0.05:
            print("The data appear to be normally distributed.")
        else:
            print("The data do not appear to be normally distributed.")

        print()
    print()
    print()


def largest_difference():
    good_df = pd.DataFrame(
        good,
        columns=[
            "Harmony",
            "Rhythm",
            "Musical Intention",
            "Subjective Evaluation",
            "Average",
        ],
    )
    bad_df = pd.DataFrame(
        bad,
        columns=[
            "Harmony",
            "Rhythm",
            "Musical Intention",
            "Subjective Evaluation",
            "Average",
        ],
    )

    # Drop 'Average' as we focus on the four main metrics
    good_df.drop("Average", axis=1, inplace=True)
    bad_df.drop("Average", axis=1, inplace=True)

    # Calculate mean differences and perform t-tests
    results = []
    for column in good_df.columns:
        good_mean = good_df[column].mean()
        bad_mean = bad_df[column].mean()
        mean_diff = good_mean - bad_mean
        stat, p_value = ttest_ind(good_df[column], bad_df[column])
        results.append((column, mean_diff, stat, p_value))

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results, columns=["Metric", "Mean Difference", "T-Statistic", "P-Value"]
    )

    # Determine the best performance metric
    best_metric = (
        results_df[results_df["P-Value"] < 0.05]
        .sort_values(by="Mean Difference", ascending=False)
        .iloc[0]
        if not results_df[results_df["P-Value"] < 0.05].empty
        else "No significant results"
    )
    print(results_df)
    print("Best performing metric:", best_metric)


def friedman_test(high, mid, low):
    for i in range(len(high)):
        data = pd.DataFrame(
            {
                "High": high[i],
                "Medium": mid[i],
                "Low": low[i],
            }
        )

        stat, p = friedmanchisquare(data["High"], data["Medium"], data["Low"])

        print(f"Friedman Test Statistic: {stat}")
        print(f"P-value: {p}")
        print()

    print()


def wilcoxon_test(high, mid, low):
    high = [item for sublist in high for item in sublist]
    mid = [item for sublist in mid for item in sublist]
    low = [item for sublist in low for item in sublist]
    data = pd.DataFrame(
        {
            "High": high,
            "Medium": mid,
            "Low": low,
        }
    )

    # Find rankings
    first, second, third = find_ranks(high, mid, low)
    names = ["High", "Medium", "Low"]

    # Pairwise comparisons
    comparisons = [(first, second), (first, third), (second, third)]
    p_values = []
    stats = []

    for idx1, idx2 in comparisons:
        stat, p_value = wilcoxon(data[names[idx1]], data[names[idx2]])
        stats.append(stat)
        p_values.append(p_value)

    # Adjust for multiple comparisons using Bonferroni correction
    p_adjusted = multipletests(p_values, alpha=0.05, method="bonferroni")

    print("Comparisons and Results:")
    for index, ((idx1, idx2), stat, p_orig, p_adj) in enumerate(
        zip(comparisons, stats, p_values, p_adjusted[1])
    ):
        print(
            f"Comparison {names[idx1]} vs {names[idx2]}: Wilcoxon test statistic = {stat}, Original P-value = {p_orig}, Adjusted P-value = {p_adj}"
        )


def find_ranks(high, mid, low):
    high_mean = mean(high)
    mid_mean = mean(mid)
    low_mean = mean(low)

    means = [(high_mean, 0), (mid_mean, 1), (low_mean, 2)]
    sorted_means = sorted(means, key=lambda x: x[0], reverse=True)

    return sorted_means[0][1], sorted_means[1][1], sorted_means[2][1]


def mean(data):
    return np.mean(data)


harmony = False
rhythm = False
musical_intention = False
subjective_evaluation = False
average = False
printers = [harmony, rhythm, musical_intention, subjective_evaluation, average]

print_pro = True
print_interest = False
print_gender = False
print_age = False

get_data()
# get_results(printers, print_pro, print_interest, print_gender, print_age)

# friedman_test(list(zip(*bad_high)), list(zip(*bad_mid)), list(zip(*bad_low)))

# wilcoxon_test(list(zip(*good_high)), list(zip(*good_mid)), list(zip(*good_low)))

# shapiro_wilk_test(list(zip(*good_low)))
# shapiro_wilk_test(list(zip(*good_mid)))
# shapiro_wilk_test(list(zip(*good_high)))

# print(t_test(pro, non_pro))
# print()
# t_test(good_mid, bad_mid)
# print()
# t_test(good_high, bad_high)

# largest_difference()
plot_figures(good, bad, "Survey Results, All Samples")


# plot_figures(good_high, bad_high, "Survey Results, High Creativity")
# plot_figures(good_mid, bad_mid, "Survey Results, Medium Creativity")
# plot_figures(good_low, bad_low, "Survey Results, Low Creativity")
