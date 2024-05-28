"""
# Hard Constraints
1) An exam will be scheduled for each course.
2) A student is enrolled in at least 3 courses. A student cannot give more than 1 exam at a time.
3) Exam will not be held on weekends. i.e if the exam is held on Saturday or Sunday, extend the slot to Monday
4) Each exam must be held between 9 am and 5 pm. For 3 slots in a day, each slot is 2 hours long.
5) Each exam must be invigilated by a teacher. A teacher cannot invigilate two exams at the same time.
6) A teacher cannot invigilate two exams in a row.

# Soft Constraints
1) All students and teachers shall be given a break on Friday from 1pm to 2pm
2) No student shall have more than 1 exams in a row
3) If a student has an MG course, schedule it before their CS course
4) Grant two hours of break in the week for faculty. Ensure half of the faculty is available at all times for invigilation. 
"""

import numpy as np
import pandas as pd


class Student:
    # Holds the name of the student and the course they are taking
    def __init__(self, name, course):
        self.name = name
        self.course = course
        self.isGivingExam = False

    def __str__(self) -> str:
        return self.name


class Course:
    # Holds the course code and course name
    def __init__(self, code, name):
        self.code = code
        self.name = name

    def __str__(self) -> str:
        res = f"Course Code: {self.code} | Course Name: {self.name}"
        return res


class Exam:
    # Holds the course, slot, day, invigilator and students taking the exam
    def __init__(self, course, slot, day, invigilator, courseCode=None, courseType=None):
        self.course = course
        self.courseCode = courseCode
        self.slot = slot
        self.day = day
        self.invigilator = invigilator
        self.isScheduled = False
        self.students = []
        self.courseType = courseType


class Schedule:
    # Holds the list of exams scheduled in their slots i.e. choromosomes for the day from 9am to 5pm. Each slot is 2 hours long with an hour break in between each exam
    # Each schedule is the full exam timetable ideally going for the highest fitness value returned from the constraints checking function.
    def __init__(
        self,
        population_size,
        mutation_rate,
        max_iterations,
        slotsPerDay=3,
        days=5,
        examDuration=2,
        examsPerSlot=5,
        classrooms=[],
    ):
        # Args
        self.population_size = population_size  # Duration of the exams in days, each schedule is a chromosome for that one day
        self.mutation_rate = mutation_rate  # Probability of mutation
        self.max_iterations = max_iterations  # Maximum number of iterations
        self.days = days  # Number of days the exams are held
        self.slotsPerDay = slotsPerDay  # Number of slots in a day
        self.examDuration = examDuration  # Duration of each exam in hours
        self.examsPerSlot = examsPerSlot  # Number of exams per slot
        self.classrooms = classrooms  # List of classrooms available for the exams

        # Privates
        self.schedule = []  # List of exams scheduled in their slots
        self.exams_data = (
            None,
        )  # Dataframe containing the courses, course codes, number of students, invigilators, slots and days scheduled for the exams
        self.student_course_data = (
            None,
        )  # Dataframe containing the students and the courses they are taking
        self.students_not_taking_exams = (
            None,
        )  # Dataframe containing the students not taking the exams
        self.teachers_data = (None,)  # Dataframe containing the teachers names
        self.fitness = 0  # Fitness value of the schedule

    def load_data(self):
        # Read data from CSV files
        courses_data = pd.read_csv(
            "./Dataset/courses.csv"
        )  # Course Code and Course Name Columns
        student_course_data = pd.read_csv(
            "./Dataset/studentCourse.csv"
        )  # Name of student and Course Code they are taking
        student_names_data = pd.read_csv(
            "./Dataset/studentNames.csv"
        )  # List of all students taking the exams or not taking the exams
        teachers_data = pd.read_csv(
            "./Dataset/teachers.csv"
        )  # Used for invigilation of exams, each teacher cannot invigilate two exams at the same time or in a row

        print(f"[LD_Data] Cleaning data...")
        # Prune courses_data to ensure every course code entry is unique
        print(
            f"[LD_Data] \tRemoved {len(courses_data) - len(courses_data.drop_duplicates(subset='Course Code'))} duplicate entries in Courses based on Course Code."
        )
        courses_data = courses_data.drop_duplicates(subset="Course Code")

        # Prune student_course_data to ensure every student-course entry is unique
        print(
            f"[LD_Data] \tRemoved {len(student_course_data) - len(student_course_data.drop_duplicates(subset=['Student Name', 'Course Code']))} duplicate entries in Student-Course based on Student Name and Course Code."
        )
        student_course_data = student_course_data.drop_duplicates(
            subset=["Student Name", "Course Code"]
        )

        # Prune student_names_data to ensure every student name entry is unique
        print(
            f"[LD_Data] \tRemoved {len(student_names_data) - len(student_names_data.drop_duplicates(subset='Names'))} duplicate entries in Student Names based on Names."
        )
        student_names_data = student_names_data.drop_duplicates(subset="Names")

        # Prune teachers_data to ensure every teacher name entry is unique
        print(
            f"[LD_Data] \tRemoved {len(teachers_data) - len(teachers_data.drop_duplicates(subset='Names'))} duplicate entries in Teachers based on Teacher Name."
        )
        teachers_data = teachers_data.drop_duplicates(subset="Names")
        print(f"[LD_Data] Data cleaned successfully.\n")

        # Create a new dataframe that has the names of all the courses available along with their course names in a seperate column
        exams_data = pd.DataFrame(
            columns=[
                "Course Code",
                "Course Name",
                "Course Type",
                "Number of Students",
                # "Invigilator",
                # "Slot",
                # "Day",
                # "Students",
            ]
        )
        # Populate the exams_data dataframe with the courses_data dataframe
        exams_data["Course Code"] = courses_data["Course Code"]
        exams_data["Course Name"] = courses_data["Course Name"]
        exams_data["Course Type"] = [
            "MG" if course_code[:2] == "MG" else "CS"
            for course_code in courses_data["Course Code"]
        ]
        exams_data["Number of Students"] = [
            len(student_course_data[student_course_data["Course Code"] == course_code])
            for course_code in courses_data["Course Code"]
        ]
        exams_data["Students Enrolled"] = [
            student_course_data[student_course_data["Course Code"] == course_code][
                "Student Name"
            ].tolist()
            for course_code in courses_data["Course Code"]
        ]

        # Sort the courses by the number of students taking the course
        exams_data = exams_data.sort_values(by="Number of Students", ascending=False)

        # Seperate the students who are not taking the exams as in students whos name is in student_names_data but not in student_course_data
        students_not_taking_exams = student_names_data[
            ~student_names_data["Names"].isin(student_course_data["Student Name"])
        ]

        # Print statments
        print(f"[LD_Data] {len(exams_data)} Exams held: \n")
        print(exams_data)
        print(
            f"\n[LD_Data] {len(students_not_taking_exams)} Students not taking exams: \n{students_not_taking_exams}"
        )

        print(f"\n[LD_Data]  Data Loaded\n")
        # Store in the class variables
        self.exams_data = exams_data
        self.student_course_data = student_course_data
        self.students_not_taking_exams = students_not_taking_exams
        self.teachers_data = teachers_data

        return exams_data, student_course_data, students_not_taking_exams, teachers_data

    def print_schedule(self):
        # Print schedule
        if not self.schedule or len(self.schedule) == 0:
            print(f"[INIT_SCHDL] Schedule is empty.")
            return
        for i, day in enumerate(self.schedule):
            print(f"[INIT_SCHDL] \tDay {i+1}: ")
            for j, slot in enumerate(day):
                print(f"[INIT_SCHDL] \t\tSlot {j+1}: ")
                for k, exam in enumerate(slot):
                    print(
                        f"[INIT_SCHDL] \t\t\t Exam {k+1}: {exam.course.code} - {exam.course.name} | {exam.invigilator} "
                    )

    def initialize_schedule(
        self,
    ):
        self.load_data()

        print(f"[INIT_SCHDL] Initializing Schedule...")
        print(
            f"[INIT_SCHDL] {self.examDuration}hr Exams will be held for {self.days} days with {self.slotsPerDay} slots per day.\n"
        )

        # Check if all exams in exams_data can be scheduled in the given number of slots, days and exams per slots
        if len(self.exams_data) > self.days * self.slotsPerDay * self.examsPerSlot:
            print(
                f"[CNSTRT ERR:] Not enough slots to schedule all exams. Increase the number of slots or days or exams per slot."
            )
            return 0
        else:
            print(
                f"[INIT_SCHDL] Available slots for the entire mids/finals are {self.days * self.slotsPerDay * self.examsPerSlot} which is higher than the number of exams which is {len(self.exams_data)}\n"
            )

        # Pick a random starting day for the exams from days. Days of exam will be as long as self.days.
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        # Create examdays list to determine the days of the exams, List will be long as self.days
        examdays = []
        for i in range(self.days):
            if i == 0:
                examdays.append(np.random.choice(days))
            else:
                examdays.append(days[(days.index(examdays[i - 1]) + 1) % len(days)])
        print(f"[HRD_CNSTRT 3/6]  Exam will not be held on the weekends")
        print(f"[INIT_SCHDL] Days of the exams: {examdays}\n")

        # Slots for the exams, each slot is an even division of time in between 9am to 6pm based on examduration. Make sure it does not exceed 5pm or 17:00
        slots = []
        for i in range(9, 17, self.slotsPerDay):
            # Throw an error if the exam duration exceeds 5pm
            if i + self.examDuration > 17:
                print(f"[CNSTRT ERR:] Exam duration exceeds 5pm")
                return 0
            slots.append(f"{i} - {i+self.examDuration}")

        print(f"[HRD_CNSTRT 4/6] Exam will be held between 9am and 5pm")
        print(f"[INIT_SCHDL] Slots for the exams: {slots}\n")

        # List to store the courses already scheduled
        scheduled_courses = []

        # List of lists of lists made by a list holding examdays, each examday has a list of slots for that day. Each slot can hold as many exams as self.examsPerSlot
        schedule = []
        for day in examdays:
            # Exam, select a random row from exams_data and populate the schedule with the exams, Randomly select an invigilator from teachers_data and ensure its unique
            day_schedule = []
            for slot in slots:
                slot_schedule = []
                # Check if all courses have been scheduled
                if len(scheduled_courses) == len(self.exams_data):
                    # If all courses have been scheduled, make the slot free by appending an empty list
                    day_schedule.append([])
                    continue
                for i in range(self.examsPerSlot):
                    # Randomly select an exam from the exams_data that hasn't been scheduled yet
                    available_courses = self.exams_data[
                        ~self.exams_data["Course Code"].isin(scheduled_courses)
                    ]
                    if available_courses.empty:
                        break  # No more courses available, exit the loop
                    exam = available_courses.sample()
                    course_code = exam["Course Code"].values[0]
                    scheduled_courses.append(course_code)

                    course = Course(course_code, exam["Course Name"].values[0])
                    # Unique invigilator for the exam
                    invigilator = np.random.choice(self.teachers_data["Names"].values)
                    while invigilator in [exam.invigilator for exam in slot_schedule]:
                        invigilator = np.random.choice(
                            self.teachers_data["Names"].values
                        )

                    # Check if it's Friday and if the slot crosses 13:00 to 14:00
                    # print(f"Day: {day}, Slot: {slot.split(' - ')[0]}")
                    start_time, end_time = slot.split(" - ")
                    if day == "Friday" and ("13:00" <= slot.split(" - ")[0] < "14:00"):
                        # Leave the slot empty
                        print(
                            f"[SOFT_CNSTRT 1/4] All students and teachers shall be given a break on Friday from 1pm to 2pm"
                        )
                        slot_schedule.append(None)
                    else:
                        # Otherwise, schedule the exam normally
                        slot_schedule.append(
                            Exam(course, slot, day, invigilator, course_code)
                        )
                day_schedule.append(slot_schedule)
            schedule.append(day_schedule)
        self.schedule = schedule

        # Print Schedule
        self.print_schedule()

        print(f"[INIT_SCHDL] Schedule initialized successfully.\n")

    def calculate_fitness(self):
        # Calculate the fitness of the schedule based on constraints being satified
        # Hard constraints are 6 in total while soft constraints are 4. The fitness value is the sum of the hard constraints and the soft constraints
        # Hard constraints have to be satisfied for the schedule to be valid, they give a fitness value of 10 each
        # Soft constraints are not necessary but they increase the fitness value of the schedule if they are satisfied, they give a fitness value of 5 each

        # Hard Constraints
        # 1) An exam will be scheduled for each course. Check if all the courses from exams_data are scheduled in the schedule
        currentFitness = 0
        print(f"\n[CLC_FTNS] Checking for Hard Constraints\n")
        for exam in self.exams_data["Course Code"]:
            if exam not in [
                exam.course.code
                for day in self.schedule
                for slot in day
                for exam in slot
            ]:
                print(
                    f"[ERR: HRD_CNSTRT 1/6] An exam will be scheduled for each course. Course {exam} not scheduled."
                )
            else:
                currentFitness += 10

        # 2) A student is enrolled in at least 3 courses. A student cannot give more than 1 exam at a time.
        # Check if a student is enrolled in at least 3 courses using the student_course_data and exams_data dataframe
        studentsNotIn3Courses = 0
        for student in self.student_course_data["Student Name"]:
            student_courses = self.student_course_data[
                self.student_course_data["Student Name"] == student
            ]["Course Code"].values
            if len(student_courses) < 3:
                print(
                    f"[ERR: HRD_CNSTRT 2/6] Student {student} is enrolled in less than 3 courses."
                )
                studentsNotIn3Courses += 1
            else:
                currentFitness += 5
        if studentsNotIn3Courses:
            print(
                f"[ERR: HRD_CNSTRT 2/6] {studentsNotIn3Courses} students are enrolled in less than 3 courses."
            )
            # Check if a student is giving more than 1 exam at a time
            for day in self.schedule:
                for slot in day:
                    student_exams = [
                        exam.course.code for exam in slot if student in exam.students
                    ]
                    if len(student_exams) > 1:
                        print(
                            f"[ERR: HRD_CNSTRT 2/6] A student cannot give more than 1 exam at a time. Student {student} is giving more than 1 exam at a time."
                        )
                    else:
                        currentFitness += 5

        # 3) Exam will not be held on weekends. i.e if the exam is held on Saturday or Sunday, extend the slot to Monday
        for day in self.schedule:
            for slot in day:
                if slot == []:
                    continue
                if slot[0].day in ["Saturday", "Sunday"]:
                    print(
                        f"[ERR: HRD_CNSTRT 3/6] Exam will not be held on weekends. Slot {slot[0].day} extended to Monday."
                    )
                else:
                    currentFitness += 10

        # 4) Each exam must be held between 9 am and 5 pm. For 3 slots in a day, each slot is 2 hours long.
        for day in self.schedule:
            for slot in day:
                for exam in slot:
                    start_time = int(exam.slot.split(" - ")[0].split(":")[0])
                    if start_time < 9 or start_time > 17:
                        print(
                            f"[ERR: HRD_CNSTRT 4/6] Each exam must be held between 9 am and 5 pm. Exam {exam.course.code} scheduled at {start_time}am."
                        )
                    else:
                        currentFitness += 10

        # 5) Each exam must be invigilated by a teacher. A teacher cannot invigilate two exams at the same time.
        for day in self.schedule:
            for slot in day:
                for exam in slot:
                    if exam.invigilator in [e.invigilator for e in slot]:
                        print(
                            f"[ERR: HRD_CNSTRT 5/6] A teacher cannot invigilate two exams at the same time. Teacher {exam.invigilator} is invigilating two exams at the same time."
                        )
                    else:
                        currentFitness += 10

        # 6) A teacher cannot invigilate two exams in a row.
        for day in self.schedule:
            for slot in day:
                for i in range(1, len(slot)):
                    if slot[i].invigilator == slot[i - 1].invigilator:
                        print(
                            f"[ERR: HRD_CNSTRT 6/6] A teacher cannot invigilate two exams in a row. Teacher {slot[i].invigilator} is invigilating two exams in a row."
                        )
                    else:
                        currentFitness += 10



        print(f"\n[CLC_FTNS] Hard Constraint Fitness Sum: {currentFitness}")

        # Soft Constraints
        print(f"\n[CLC_FTNS] Checking for Soft Constraints\n")
        # 1) All students and teachers shall be given a break on Friday from 1pm to 2pm
        for day in self.schedule:
            for slot in day:
                for exam in slot:
                    if exam.day == "Friday" and "13:00" <= exam.slot.split(" - ")[0] < "14:00":
                        print(
                            f"[SOFT_CNSTRT 1/4] All students and teachers shall be given a break on Friday from 1pm to 2pm"
                        )
                        currentFitness += 5

        # 2) No student shall have more than 1 exams in a row. Check if slot is empty that means theres no student so do not throw an error or count it as a fitness increase
        for day in self.schedule:
            for slot in day:
                if slot == []:
                    continue
                for i in range(1, len(slot)):
                    if slot[i].students == slot[i - 1].students:
                        print(
                            f"[SOFT_CNSTRT 2/4] No student shall have more than 1 exams in a row. Student {slot[i].students} has more than 1 exam in a row."
                        )
                    else:
                        currentFitness += 5

        # 3) If a student has an MG course, schedule it before their CS course
        for day in self.schedule:
            for slot in day:
                for exam in slot:
                    if exam.courseType == "MG":
                        print(
                            f"[SOFT_CNSTRT 3/4] If a student has an MG course, schedule it before their CS course"
                        )
                        currentFitness += 5

        # 4) Grant two hours of break in the week for faculty. Ensure half of the faculty is available at all times for invigilation.
        for day in self.schedule:
            for slot in day:
                for exam in slot:
                    if exam.slot.split(" - ")[0] == "13:00":
                        print(
                            f"[SOFT_CNSTRT 4/4] Grant two hours of break in the week for faculty. Ensure half of the faculty is available at all times for invigilation."
                        )
                        currentFitness += 5

        print(f"\n[CLC_FTNS] Soft Constraint Fitness Sum: {currentFitness}")


    def onePointCrossover(self, parent1, parent2):
        # One point crossover
        print(f"[CRS_OPR] Performing One Point Crossover")
        crossoverPoint = np.random.randint(1, len(parent1))
        print(f"[CRS_OPR] Crossover Point: \"[:{crossoverPoint}]\"")
        child1 = parent1[:crossoverPoint] + parent2[crossoverPoint:]
        child2 = parent2[:crossoverPoint] + parent1[crossoverPoint:]
        print(f"[CRS_OPR] Child 1: {child1}")
        print(f"[CRS_OPR] Child 2: {child2}")

        return child1, child2
    
    def swapMutation(self, child):
        # Swap two rows of the schedule
        print(f"[MUTATION] Performing Swap Mutation")
        mutationPoint1 = np.random.randint(1, len(child))
        mutationPoint2 = np.random.randint(1, len(child))
        print(f"[MUTATION] Mutation Points: {mutationPoint1}, {mutationPoint2}")
        child[mutationPoint1], child[mutationPoint2] = child[mutationPoint2], child[mutationPoint1]
        print(f"[MUTATION] Child: {child}")

        return child

    def GeneticAlgorithm(self):
        # Genetic Algorithm
        print(f"[GN_ALGO] Running Genetic Algorithm")
        self.initialize_schedule()
        self.calculate_fitness()

        # Initialize the population
        population = self.initialize_population()

        # Perform the genetic algorithm
        for i in range(self.max_iterations):
            print(f"\n[GN_ALGO] Iteration {i+1}\n")
            # Calculate the fitness of the population
            fitness = self.calculate_fitness()

            # Select the best parents
            parents = self.select_parents(population)

            # Perform crossover
            children = []
            for j in range(0, len(parents), 2):
                child1, child2 = self.onePointCrossover(parents[j], parents[j+1])
                children.append(child1)
                children.append(child2)

            # Perform mutation
            for child in children:
                if np.random.random() < self.mutation_rate:
                    child = self.swapMutation(child)

            # Select the best children
            population = self.select_population(population, children)

            # Calculate the fitness of the population
            fitness = self.calculate_fitness()

            # Print the best fitness value
            print(f"[GN_ALGO] Best Fitness Value: {fitness}")

    def initialize_population(self):
        # Initialize the population
        print(f"[INIT_POP] Initializing Population")
        population = []
        for i in range(self.population_size):
            # Shuffle the schedule
            np.random.shuffle(self.schedule)
            population.append(self.schedule)
        print(f"[INIT_POP] Population Initialized")



        return population
    
    def select_parents(self, population):
        # Select the best parents
        print(f"[SLCT_PRNT] Selecting Parents")
        parents = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i+1]
            if self.calculate_fitness(parent1) > self.calculate_fitness(parent2):
                parents.append(parent1)
            else:
                parents.append(parent2)
        print(f"[SLCT_PRNT] Parents Selected")
        return parents

if __name__ == "__main__":
    try:

        classrooms = [
            "C301",
            "C302",
            "C303",
            "C304",
            "C305",
            "C306",
            "C307",
            "C308",
            "C309",
            "C310",
            "C311",
            "C312",
            "C313",
            "C314",
            "C315",
        ]
        classroomsToUse = classrooms[:10]

        schedule = Schedule(
            population_size=100,
            mutation_rate=0.1,
            max_iterations=100,
            slotsPerDay=3,
            examDuration=2,
            days=3,
            classrooms=classroomsToUse,
        )

        schedule.GeneticAlgorithm()
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C. Exiting...")
