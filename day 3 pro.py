import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)

########
data = {
    "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Name": ["Rahul", "Madhumita", "Sara", "Amit", "Priya", "John", "Riya", "Arjun"],
    "Gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male"],
    "Math": [85, 92, 78, np.nan, 88, 76, 95, 67],
    "Science": [90, 89, np.nan, 72, 85, 80, 98, 70],
    "English": [88, 94, 79, 75, np.nan, 82, 96, 68],
    "Attendance": [92, 96, 85, 78, 88, 80, 98, np.nan],
    "Study_Hours": [3.5, 5.0, 2.5, 2.0, 4.0, 3.0, 6.0, 1.5]
}

df = pd.DataFrame(data)

######
print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== SHAPE OF DATASET ==========")
print("Rows, Columns:", df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

#########
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

########
df["Math"].fillna(df["Math"].mean(), inplace=True)
df["Science"].fillna(df["Science"].mean(), inplace=True)
df["English"].fillna(df["English"].mean(), inplace=True)
df["Attendance"].fillna(df["Attendance"].mean(), inplace=True)

print("\n========== DATASET AFTER FILLING MISSING VALUES ==========")
print(df)

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

##########

df["Total"] = df["Math"] + df["Science"] + df["English"]
df["Average"] = df["Total"] / 3

# Grade calculation
def assign_grade(avg):
    if avg >= 90:
        return "A+"
    elif avg >= 80:
        return "A"
    elif avg >= 70:
        return "B"
    elif avg >= 60:
        return "C"
    else:
        return "D"

df["Grade"] = df["Average"].apply(assign_grade)

print("\n========== DATASET WITH TOTAL, AVERAGE, GRADE ==========")
print(df)

######
print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

#####

topper = df.loc[df["Average"].idxmax()]

print("\n========== TOPPER STUDENT ==========")
print("Name:", topper["Name"])
print("Average Marks:", round(topper["Average"], 2))
print("Grade:", topper["Grade"])
#####
high_attendance = df[df["Attendance"] > 90]

print("\n========== STUDENTS WITH ATTENDANCE > 90 ==========")
print(high_attendance[["Name", "Attendance"]])
####
high_performers = df[df["Average"] > 80]

print("\n========== STUDENTS WITH AVERAGE > 80 ==========")
print(high_performers[["Name", "Average", "Grade"]])
####
gender_avg = df.groupby("Gender")[["Math", "Science", "English", "Average"]].mean()

print("\n========== AVERAGE MARKS BY GENDER ==========")
print(gender_avg)
#####
sorted_df = df.sort_values(by="Average", ascending=False)

print("\n========== STUDENTS SORTED BY AVERAGE MARKS ==========")
print(sorted_df[["Name", "Average", "Grade"]])
#####
plt.figure(figsize=(10, 5))
plt.bar(df["Name"], df["Average"])
plt.title("Student Average Marks")
plt.xlabel("Student Name")
plt.ylabel("Average Marks")
plt.xticks(rotation=45)
plt.show()
     #####
plt.figure(figsize=(10, 5))
plt.plot(df["Name"], df["Attendance"], marker='o')
plt.title("Student Attendance")
plt.xlabel("Student Name")
plt.ylabel("Attendance")
plt.xticks(rotation=45)
plt.show()

#######
plt.figure(figsize=(8, 5))
plt.hist(df["Average"], bins=5, edgecolor="black")
plt.title("Distribution of Average Marks")
plt.xlabel("Average Marks")
plt.ylabel("Frequency")
plt.show()
#####
grade_counts = df["Grade"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(grade_counts, labels=grade_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Grade Distribution")
plt.show()
#####
plt.figure(figsize=(8, 5))
sns.countplot(x="Grade", data=df)
plt.title("Count of Students by Grade")
plt.show()
     
####
plt.figure(figsize=(8, 5))
sns.boxplot(x="Gender", y="Average", data=df)
plt.title("Average Marks by Gender")
plt.show()
####
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Study_Hours", y="Average", hue="Gender", data=df, s=100)
plt.title("Study Hours vs Average Marks")
plt.show()
     
     ###
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
####
df.to_csv("cleaned_student_performance.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_student_performance.csv'")
###
print("\n========== FINAL CLEANED DATASET ==========")
print(df)
