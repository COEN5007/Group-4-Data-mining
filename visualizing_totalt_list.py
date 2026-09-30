import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("total_list.csv")

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


columns = [
    "sms_convo_S-H",
    "bt_interactions_S-H",
    "calls_duration_S-H",
    "fb_friends"
]



data["bt_interactions_S-H"] = (
    data["bt_interactions_S-H"]
    .astype(str)
    .str.replace("−", "-", regex=False)
)

data["bt_interactions_S-H"] = pd.to_numeric(
    data["bt_interactions_S-H"],
    errors="coerce"
)

print(data["bt_interactions_S-H"].dtype)
print(data["bt_interactions_S-H"].unique())



columns = [
    "sms_convo_S-H",
    "bt_interactions_S-H",
    "calls_duration_S-H",
    "fb_friends"
]

plt.figure(figsize=(10, 6))

plt.boxplot(
    [data[col].dropna() for col in columns],
    tick_labels=[
        "SMS conversations",
        "Bluetooth interactions",
        "Call duration",
        "Facebook friends"
    ]
)

plt.ylabel("Value")
plt.title("Distribution and outliers in features")

plt.show()