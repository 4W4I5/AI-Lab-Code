import pandas as pd

# Assuming 'self.student_course_data' is the DataFrame containing the dataset
student_course_data = pd.read_csv(
    "./Dataset/studentCourse.csv"
)  # Name of student and Course Code they are taking
student_course_count = student_course_data.groupby('Student Name')['Course Code'].count()

# Convert the series to a DataFrame and sort by the count of courses in descending order
sorted_student_course_count = student_course_count.reset_index(name='Count of Courses').sort_values(by='Count of Courses', ascending=False)

# Print the sorted count of courses for each student
for index, row in sorted_student_course_count.iterrows():
    print(f"{row['Student Name']}: {row['Count of Courses']} courses")