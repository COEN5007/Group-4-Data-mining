import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("Final_total_list.csv")

print(data.shape)
print(data.head(10))
print(data.describe())


# creating a histogram

#genders 
"""plt.hist(data['gender'])
plt.title('Deviation of gender among the students')
plt.xlabel('1 = female, 0 = male')
plt.ylabel('students amt.')
plt.show()
"""
#facebook friends 
"""plt.hist(data['fb_friends'])
plt.title('FB friends')
plt.xlabel('friends amt.')
plt.ylabel('students amt.')
plt.show()"""

# school hours sms sent
"""plt.hist(data['sms_sent_S-H'])
plt.title(' school hours')
plt.xlabel('sms sent')
plt.ylabel('students amt.')
plt.show()"""


"""plt.hist(data['sms_sent_O-S-H'])
plt.title('Off school hours')
plt.xlabel('duration')
plt.ylabel('students amt.')
plt.show()"""


"""columns = [
    "sms_convo_S-H",
    "bt_interactions_S-H",
    "calls_duration_S-H",
    "fb_friends"
]


for column in data.columns:

    # user är bara ett ID
    if column == "user":
        continue

    plt.figure()

    plt.boxplot(data[column].dropna())

    plt.title(column)
    plt.ylabel(column)

    plt.show()
"""

"""import matplotlib.pyplot as plt
import seaborn as sns

corr_data = data.drop(columns=["user", "gender"])
correlation = corr_data.corr()

plt.figure(figsize=(22, 18))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    square=True
)

plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.title("Correlation between features", fontsize=16)

plt.tight_layout()
plt.show()"""


"""print("Mean:", data["sms_convo_O-S-H"].mean())
print("Median:", data["sms_convo_O-S-H"].median())
print("Max:", data["sms_convo_O-S-H"].max())
print("Lower quartile:", data["sms_convo_O-S-H"].())
print("övre kvartil:", data["sms_convo_O-S-H"].max())

#vill ha detta i en tabell
"""


#tabell till excel fil

analysis_data = data.drop(columns=["user"])

stats = pd.DataFrame({
    "Mean": analysis_data.mean(),
    "Median": analysis_data.median(),
    "Max": analysis_data.max(),
    "Lower quartile (Q1)": analysis_data.quantile(0.25),
    "Upper quartile (Q3)": analysis_data.quantile(0.75)
})

print(stats)

# tar bort user eftersom det bara är ett ID
analysis_data = data.drop(columns=["user"])

# skapar tabellen
stats = pd.DataFrame({
    "Min": analysis_data.min(),
    "Lower quartile (Q1)": analysis_data.quantile(0.25),
    "Median": analysis_data.median(),
    "Mean": analysis_data.mean(),
    "Upper quartile (Q3)": analysis_data.quantile(0.75),
    "Max": analysis_data.max(),
    "Standard deviation": analysis_data.std()
})

# avrundar till 2 decimaler
stats = stats.round(2)

# sparar som Excel
stats.to_excel("descriptive_statistics.xlsx")

print("Excel file created!")

