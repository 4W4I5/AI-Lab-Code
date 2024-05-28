import random
from datetime import timedelta

import numpy as np
import pandas as pd


class Student:
    def __init__(self, name, course):
        self.name = name
        self.course = course
        self.isGivingExam = False

    def __str__(self) -> str:
        return self.name


class Course:
    def __init__(self, code, name):
        self.code = code
        self.name = name

    def __str__(self) -> str:
        return f"Course Code: {self.code} | Course Name: {self.name}"


class Exam:
    def __init__(
        self, course, slot, day, invigilator, courseCode=None, courseType=None
    ):
        self.course = course
        self.courseCode = courseCode
        self.slot = slot
        self.day = day
        self.invigilator = invigilator
        self.isScheduled = False
        self.students = []
        self.courseType = courseType


class Schedule:
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
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_iterations = max_iterations
        self.days = days
        self.slotsPerDay = slotsPerDay
        self.examDuration = examDuration
        self.examsPerSlot = examsPerSlot
        self.classrooms = classrooms

        self.schedule = []
        self.exams_data = None
        self.student_course_data = None
        self.students_not_taking_exams = None
        self.teachers_data = None
        self.fitness = 0

    def print_schedule(self):
        # Print schedule
        if not self.schedule or len(self.schedule) == 0:
            print("[INIT_SCHDL] Schedule is empty.")
            return
        for i, day in enumerate(self.schedule):
            print(f"[INIT_SCHDL] Day {i+1}: ")
            for j, slot in enumerate(day):
                print(f"[INIT_SCHDL] \tSlot {j+1}: ")
                if not slot:
                    print("[INIT_SCHDL] \t\tNo exams scheduled in this slot.")
                else:
                    for k, exam in enumerate(slot):
                        if exam is None:
                            print("[INIT_SCHDL] \t\t\tNo exam scheduled in this slot.")
                        else:
                            print(
                                f"[INIT_SCHDL] \t\t\tExam {k+1}: {exam.course.code} - {exam.course.name} | {exam.invigilator} "
                            )

    def load_data(self):
        # Load data from CSV files
        courses_data = pd.read_csv("./Dataset/courses.csv")
        student_course_data = pd.read_csv("./Dataset/studentCourse.csv")
        student_names_data = pd.read_csv("./Dataset/studentNames.csv")
        teachers_data = pd.read_csv("./Dataset/teachers.csv")

        # Data cleaning and preprocessing
        # Remove duplicate entries
        courses_data.drop_duplicates(subset="Course Code", inplace=True)
        student_course_data.drop_duplicates(
            subset=["Student Name", "Course Code"], inplace=True
        )
        student_names_data.drop_duplicates(subset="Names", inplace=True)
        teachers_data.drop_duplicates(subset="Names", inplace=True)

        # Create a new dataframe to store exam data
        exams_data = pd.DataFrame(
            columns=[
                "Course Code",
                "Course Name",
                "Course Type",
                "Number of Students",
                "Students Enrolled",
            ]
        )

        # Populate the exams_data dataframe with information from other datasets
        exams_data["Course Code"] = courses_data["Course Code"]
        exams_data["Course Name"] = courses_data["Course Name"]
        exams_data["Course Type"] = exams_data["Course Code"].apply(
            lambda x: "MG" if x.startswith("MG") else "CS"
        )
        exams_data["Number of Students"] = (
            student_course_data.groupby("Course Code")
            .size()
            .reset_index(name="Count")["Count"]
        )
        exams_data["Students Enrolled"] = (
            student_course_data.groupby("Course Code")["Student Name"]
            .apply(list)
            .reset_index()["Student Name"]
        )

        # Sort exams_data by the number of students enrolled in each course
        exams_data.sort_values(by="Number of Students", ascending=False, inplace=True)

        # Identify students not taking exams
        students_not_taking_exams = student_names_data[
            ~student_names_data["Names"].isin(student_course_data["Student Name"])
        ]

        # Print summary
        print("[LOAD_DATA] Data loaded successfully.")
        print(f"[LOAD_DATA] Exams data: {len(exams_data)} courses")
        print(f"[LOAD_DATA] Student-course data: {len(student_course_data)} entries")
        print(
            f"[LOAD_DATA] Students not taking exams: {len(students_not_taking_exams)} students"
        )
        print(f"[LOAD_DATA] Teachers data: {len(teachers_data)} teachers")

        # Assign data to class variables
        self.exams_data = exams_data
        self.student_course_data = student_course_data
        self.students_not_taking_exams = students_not_taking_exams
        self.teachers_data = teachers_data

    def initialize_schedule(self):
        self.load_data()

        print("[INIT_SCHDL] Initializing Schedule...")
        print(
            f"[INIT_SCHDL] {self.examDuration}hr Exams will be held for {self.days} days with {self.slotsPerDay} slots per day.\n"
        )

        # Check if there are enough slots to schedule all exams
        total_exams = len(self.exams_data)
        total_slots = self.days * self.slotsPerDay * self.examsPerSlot
        if total_exams > total_slots:
            print(
                f"[INIT_SCHDL] Error: Not enough slots to schedule all exams. Increase the number of slots or days or exams per slot."
            )
            return

        # Define the days of the exams
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        exam_days = [days[i % len(days)] for i in range(self.days)]
        print(f"[INIT_SCHDL] Days of the exams: {exam_days}\n")

        # Define the time slots for exams
        slots = []
        for i in range(9, 17, self.slotsPerDay):
            # Throw an error if the exam duration exceeds 5pm
            if i + self.examDuration > 17:
                print(f"[CNSTRT ERR:] Exam duration exceeds 5pm")
                return 0
            slots.append(f"{i} - {i+self.examDuration}")
        print(f"[INIT_SCHDL] Slots for the exams: {slots}\n")

        # Initialize schedule as a list of lists of lists to hold exams
        schedule = []
        for day in exam_days:
            day_schedule = []
            for slot in slots:  # Iterate over slots instead of using range
                slot_schedule = []
                for _ in range(self.examsPerSlot):
                    if total_exams > 0:
                        # Randomly select an exam from exams_data
                        exam = self.exams_data.iloc[total_exams - 1]
                        course = Course(exam["Course Code"], exam["Course Name"])
                        invigilator = np.random.choice(self.teachers_data["Names"])
                        exam_object = Exam(
                            course,
                            slot,  # Pass the slot value here
                            day,
                            invigilator,
                            exam["Course Code"],
                            exam["Course Type"],
                        )
                        slot_schedule.append(exam_object)
                        total_exams -= 1
                    else:
                        slot_schedule.append(None)
                day_schedule.append(slot_schedule)
            schedule.append(day_schedule)

        self.schedule = schedule

        # Print schedule
        print("[INIT_SCHDL] Schedule initialized successfully.\n")
        self.print_schedule()

    def initialize_population(self):
        # Initialize the population
        print("[INIT_POP] Initializing Population")
        population = []
        for i in range(self.population_size):
            # Shuffle the schedule
            shuffled_schedule = self.schedule.copy()
            for day in shuffled_schedule:
                np.random.shuffle(day)
            population.append(shuffled_schedule)
        print("[INIT_POP] Population Initialized")
        return population

    def calculate_fitness(self, schedule=None):
        if schedule is None:
            schedule = self.schedule

        # Initialize fitness value
        fitness = 0

        # Hard Constraints
        print("[CLC_FTNS] Checking for Hard Constraints\n")
        # 1) An exam will be scheduled for each course.
        scheduled_courses = [
            exam.course.code
            for day in schedule
            for slot in day
            for exam in slot
            if exam
        ]
        if set(scheduled_courses) == set(self.exams_data["Course Code"]):
            fitness += 10
        else:
            print(
                "[ERR: HRD_CNSTRT 1/6] An exam will be scheduled for each course. Some courses are not scheduled."
            )

        # 2) A student is enrolled in at least 3 courses. A student cannot give more than 1 exam at a time.
        students_courses_count = self.student_course_data.groupby("Student Name")[
            "Course Code"
        ].count()
        if (students_courses_count >= 3).all() and not any(students_courses_count > 1):
            fitness += 10
        else:
            print(
                "[ERR: HRD_CNSTRT 2/6] A student is enrolled in at least 3 courses. Some students are not meeting this requirement."
            )

        # 3) Exam will not be held on weekends. Extend the slot to Monday if the exam is held on Saturday or Sunday.
        weekend_days = ["Saturday", "Sunday"]
        if any(
            exam.day in weekend_days
            for day in schedule
            for slot in day
            for exam in slot
            if exam
        ):
            print(
                "[ERR: HRD_CNSTRT 3/6] Exam will not be held on weekends. Some exams are scheduled on weekends."
            )
        else:
            fitness += 10

        # 4) Each exam must be held between 9 am and 5 pm.
        if all(
            9 <= int(exam.slot.split(" - ")[0].split(":")[0]) <= 16
            for day in schedule
            for slot in day
            for exam in slot
            if exam
        ):
            fitness += 10
        else:
            print(
                "[ERR: HRD_CNSTRT 4/6] Each exam must be held between 9 am and 5 pm. Some exams are scheduled outside this time range."
            )

        # 5) Each exam must be invigilated by a teacher.
        invigilators = [
            exam.invigilator
            for day in schedule
            for slot in day
            for exam in slot
            if exam
        ]
        if len(invigilators) == len(set(invigilators)):
            fitness += 10
        else:
            print(
                "[ERR: HRD_CNSTRT 5/6] Each exam must be invigilated by a teacher. Some teachers are invigilating multiple exams at the same time."
            )

        # 6) A teacher cannot invigilate two exams in a row.
        for day in schedule:
            for slot_index in range(len(day) - 1):
                current_slot = day[slot_index]
                next_slot = day[slot_index + 1]
                current_invigilators = [
                    exam.invigilator for exam in current_slot if exam
                ]
                next_invigilators = [exam.invigilator for exam in next_slot if exam]
                if any(
                    invigilator in next_invigilators
                    for invigilator in current_invigilators
                ):
                    print(
                        "[ERR: HRD_CNSTRT 6/6] A teacher cannot invigilate two exams in a row."
                    )
                    break
            else:
                fitness += 10

        print(f"\n[CLC_FTNS] Hard Constraint Fitness Sum: {fitness}\n")

        # Soft Constraints
        print("[CLC_FTNS] Checking for Soft Constraints\n")
        # 1) All students and teachers shall be given a break on Friday from 1pm to 2pm
        if any(
            exam.day == "Friday" and "13:00" <= exam.slot.split(" - ")[0] < "14:00"
            for day in schedule
            for slot in day
            for exam in slot
            if exam
        ):
            fitness += 5
        else:
            print(
                "[SOFT_CNSTRT 1/4] All students and teachers shall be given a break on Friday from 1pm to 2pm."
            )

        # 2) No student shall have more than 1 exam in a row.
        for day in schedule:
            for slot_index in range(len(day) - 1):
                current_slot = day[slot_index]
                next_slot = day[slot_index + 1]
                current_students = [exam.students for exam in current_slot if exam]
                next_students = [exam.students for exam in next_slot if exam]
                if any(student in next_students for student in current_students):
                    print(
                        "[SOFT_CNSTRT 2/4] No student shall have more than 1 exam in a row."
                    )
                    break
            else:
                fitness += 5

        # 3) If a student has an MG course, schedule it before their CS course.
        for day in schedule:
            for slot in day:
                mg_courses = [
                    exam.course.code
                    for exam in slot
                    if exam and exam.courseType == "MG"
                ]
                cs_courses = [
                    exam.course.code
                    for exam in slot
                    if exam and exam.courseType == "CS"
                ]
                if mg_courses and cs_courses and mg_courses[-1] > cs_courses[0]:
                    print(
                        "[SOFT_CNSTRT 3/4] If a student has an MG course, schedule it before their CS course."
                    )
                    break
            else:
                fitness += 5

        # 4) Grant two hours of break in the week for faculty. Ensure half of the faculty is available at all times for invigilation.
        for day in schedule:
            for slot in day:
                if any(exam.slot.split(" - ")[0] == "13:00" for exam in slot if exam):
                    print(
                        "[SOFT_CNSTRT 4/4] Grant two hours of break in the week for faculty. Ensure half of the faculty is available at all times for invigilation."
                    )
                    break
            else:
                fitness += 5

        print(f"\n[CLC_FTNS] Soft Constraint Fitness Sum: {fitness}\n")

        return fitness

    def one_point_crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)

        # Perform crossover
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def swapMutation(self, child):
        # Implement mutation operator to introduce variation in the population
        mutationPoint1 = random.randint(0, len(child) - 1)
        mutationPoint2 = random.randint(0, len(child) - 1)

        # Perform mutation
        child[mutationPoint1], child[mutationPoint2] = (
            child[mutationPoint2],
            child[mutationPoint1],
        )

        return child

    def select_parents(self, population):
        # Select the best parents
        print("[SLCT_PRNT] Selecting Parents")
        parents = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            if self.calculate_fitness(parent1) > self.calculate_fitness(parent2):
                parents.append(parent1)
            else:
                parents.append(parent2)
        print("[SLCT_PRNT] Parents Selected")
        return parents

    def select_population(self, population, children):
        # Combine the current population and the new children
        combined_population = population + children

        # Sort the combined population by fitness in descending order
        sorted_population = sorted(
            combined_population, key=lambda x: self.calculate_fitness(x), reverse=True
        )

        # Select the top individuals from the combined population
        selected_population = sorted_population[: self.population_size]

        return selected_population

    def GeneticAlgorithm(self):
        # Run the genetic algorithm to optimize the exam schedule
        # Genetic Algorithm
        print("[GN_ALGO] Running Genetic Algorithm")
        self.initialize_schedule()
        self.calculate_fitness()

        # Initialize the population
        population = self.initialize_population()
        last_generation = population.copy()

        # Perform the genetic algorithm
        for i in range(self.max_iterations):
            print(f"\n[GN_ALGO] Iteration {i+1}\n")
            # Calculate the fitness of the population
            fitness = self.calculate_fitness()

            # Select the best parents from the last two generations
            parents = self.select_parents(last_generation)

            # Perform crossover
            children = []
            for j in range(0, len(parents), 2):
                child1, child2 = self.one_point_crossover(parents[j], parents[j + 1])
                children.append(child1)
                children.append(child2)

            # Perform mutation
            for child in children:
                if random.random() < self.mutation_rate:
                    child = self.swapMutation(child)

            # Select the best children
            population = self.select_population(population, children)
            last_generation = (
                population.copy()
            )  # Update the last generation for parent selection

            # Calculate the fitness of the population
            fitness = self.calculate_fitness()

            # Print the best fitness value
            print(f"[GN_ALGO] Best Fitness Value: {fitness}")

        # Print the final exam schedule
        print("\n[GN_ALGO] Final Exam Schedule\n")
        self.print_schedule()


if __name__ == "__main__":
    try:
        # Initialize parameters and create a Schedule instance
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
            max_iterations=5,
            slotsPerDay=3,
            examDuration=2,
            days=3,
            classrooms=classroomsToUse,
        )

        # Run the genetic algorithm to generate an optimized exam schedule
        schedule.GeneticAlgorithm()
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C. Exiting...")
