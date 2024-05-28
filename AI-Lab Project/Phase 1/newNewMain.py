class Schedule:
    def __init__(
        self,
        population_size,
        mutation_rate,
        max_iterations,
        slotsPerDay=3,
        days=5,
        examsPerSlot=3,
        examDuration=2,
        classrooms=[],
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_iterations = max_iterations
        self.days = days
        self.slotsPerDay = slotsPerDay
        self.examsPerSlot = examsPerSlot
        self.classrooms = classrooms
        self.schedule = []
        self.exams_data = None
        self.examDuration = examDuration
        self.examDays = None
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
                            if exam.classroom:
                                print(
                                    f"[INIT_SCHDL] \t\t\tExam {k+1}: {exam.course.code} - {exam.course.name} | {exam.invigilator} | Classroom: {exam.classroom}"
                                )
                            else:
                                print(
                                    f"[INIT_SCHDL] \t\t\tExam {k+1}: {exam.course.code} - {exam.course.name} | {exam.invigilator} | Classroom not assigned"
                                )

    def load_data(self):
        # Load data from CSV files
        courses_data = pd.read_csv("./Dataset/courses.csv")
        student_course_data = pd.read_csv("./Dataset/studentCourse.csv")
        student_names_data = pd.read_csv("./Dataset/studentNames.csv")
        teachers_data = pd.read_csv("./Dataset/teachers.csv")

        # Data cleaning and preprocessing
        courses_data.drop_duplicates(subset="Course Code", inplace=True)
        student_course_data.drop_duplicates(
            subset=["Student Name", "Course Code"], inplace=True
        )
        student_names_data.drop_duplicates(subset="Names", inplace=True)
        teachers_data.drop_duplicates(subset="Names", inplace=True)

        # Count the number of courses each student is enrolled in
        student_courses_count = (
            student_course_data.groupby("Student Name")["Course Code"]
            .count()
            .reset_index(name="Course Count")
        )

        # Keep only students enrolled in 3 or more courses
        student_course_data = pd.merge(
            student_course_data,
            student_courses_count[student_courses_count["Course Count"] >= 3],
            on="Student Name",
        )

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
            f"[INIT_SCHDL] {self.examDuration}-hour Exams will be held for {self.days} days with {self.slotsPerDay} slots per day.\n"
        )

        # Check if there are enough slots to schedule all exams
        total_exams = len(self.exams_data)
        total_slots = self.days * self.slotsPerDay * self.examsPerSlot
        if total_exams > total_slots:
            print(
                "[INIT_SCHDL] Error: Not enough slots to schedule all exams. Increase the number of slots or days or exams per slot."
            )
            return

        # Define the days of the exams
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        exam_days = [days[i % len(days)] for i in range(self.days)]
        self.examDays = exam_days
        print(f"[INIT_SCHDL] Days of the exams: {exam_days}\n")

        # Define the time slots for exams
        slots = []
        for i in range(9, 17, self.slotsPerDay):
            # Throw an error if the exam duration exceeds 5pm
            if i + self.examDuration > 17:
                print(f"[CNSTRT ERR:] Exam duration exceeds 5pm")
                return 0
            slots.append(f"{i} - {i+self.examDuration}")

        # Initialize schedule as a list of lists of lists to hold exams
        schedule = []
        for day in exam_days:
            day_schedule = []
            for slot in slots:
                slot_schedule = []
                for _ in range(self.examsPerSlot):
                    if total_exams > 0:
                        exam = self.exams_data.iloc[total_exams - 1]
                        course = Course(exam["Course Code"], exam["Course Name"])
                        invigilator = np.random.choice(self.teachers_data["Names"])
                        exam_object = Exam(
                            course,
                            slot,
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

        # Assign classrooms to exams
        available_classrooms_per_day = {
            day: self.classrooms.copy() for day in exam_days
        }
        for day_index, day in enumerate(schedule):
            for slot_index, slot in enumerate(day):
                for exam in slot:
                    if exam:
                        if available_classrooms_per_day[exam.day]:
                            exam.classroom = available_classrooms_per_day[exam.day].pop(
                                0
                            )
                        else:
                            print(
                                "Error: Not enough classrooms to assign to all exams for",
                                exam.day,
                            )
                            return

        self.schedule = schedule

        # Print schedule
        print("[INIT_SCHDL] Schedule initialized successfully.\n")
        self.print_schedule()

    def initialize_population(self):
        print("[INIT_POP] Initializing Population")
        population = []
        for _ in range(self.population_size):
            shuffled_schedule = self.schedule.copy()
            for day in shuffled_schedule:
                np.random.shuffle(day)
            population.append(shuffled_schedule)
        print("[INIT_POP] Population Initialized")
        return population

    def calculate_fitness(self, schedule=None):
        if schedule is None:
            schedule = self.schedule

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

        # 2) A student is enrolled in at least 3 courses.
        students_courses_count = self.student_course_data.groupby("Student Name")[
            "Course Code"
        ].count()
        if (students_courses_count >= 3).all():
            fitness += 10
        else:
            print(
                "[ERR: HRD_CNSTRT 2/6] A student is enrolled in at least 3 courses. Some students are not meeting this requirement."
            )

        # 3) Exam will not be held on weekends.
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
            9 <= int(exam.slot.split(" - ")[0].split(":")[0]) < 17
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
        if all(exam.invigilator for day in schedule for slot in day for exam in slot if exam):
            fitness += 10
        else:
            print(
                "[ERR: HRD_CNSTRT 5/6] Each exam must be invigilated by a teacher. Some exams do not have an invigilator assigned."
            )

        # 6) A teacher cannot invigilate two exams in a row. Ignore empty slots
        for day in schedule:
            for i in range(len(day) - 1):
                if all(day[i]) and all(day[i + 1]):
                    if day[i][-1].invigilator == day[i + 1][0].invigilator:
                        print(
                            "[ERR: HRD_CNSTRT 6/6] A teacher cannot invigilate two exams in a row. A teacher is invigilating two exams in a row."
                        )
                        break
            else:
                fitness += 10

        print(f"\n[CLC_FTNS] Hard Constraint Fitness Sum: {fitness}\n")

        # Soft Constraints
        print("[CLC_FTNS] Checking for Soft Constraints\n")

        # Soft Constraint 1: All students and teachers shall be given a break on Friday from 1pm to 2pm
        for day in schedule:
            for slot in day:
                for exam in slot:
                    if exam and exam.day == "Friday" and exam.slot == "13:00 - 14:00":
                        print(
                            "[ERR: SFT_CNSTRT 1/4] All students and teachers shall be given a break on Friday from 1pm to 2pm. An exam is scheduled during this time."
                        )
                        break
            else:
                fitness += 5

        # Soft Constraint 2: No student shall have more than 1 exam in a row
        for student in self.student_course_data["Student Name"].unique():
            exams_schedule = [
                exam
                for day in schedule
                for slot in day
                for exam in slot
                if exam and student in [student.name for student in exam.students]
            ]
            for i in range(len(exams_schedule) - 1):
                if exams_schedule[i].day == exams_schedule[i + 1].day:
                    if (
                        exams_schedule[i].slot.split(" - ")[1]
                        == exams_schedule[i + 1].slot.split(" - ")[0]
                    ):
                        print(
                            "[ERR: SFT_CNSTRT 2/4] No student shall have more than 1 exam in a row. A student has exams scheduled back-to-back."
                        )
                        break
            else:
                fitness += 5

        # Soft Constraint 3: If a student has an MG course, award points if it is scheduled before their CS course. use the exams_data dataframe to get the course type
        for student in self.student_course_data["Student Name"].unique():
            exams_schedule = [
                exam
                for day in schedule
                for slot in day
                for exam in slot
                if exam and student in [student.name for student in exam.students]
            ]
            for i in range(len(exams_schedule) - 1):
                if exams_schedule[i].courseType == "MG":
                    if exams_schedule[i + 1].courseType == "CS":
                        print(
                            "[ERR: SFT_CNSTRT 3/4] If a student has an MG course, award points if it is scheduled before their CS course. A student has an MG course scheduled after their CS course."
                        )
                        break
            else:
                fitness += 5

        # Soft Constraint 4: Grant two hours of break in the week for faculty. Ensure half of the faculty is available at all times for invigilation.
        faculty_available = []
        for day in schedule:
            for slot in day:
                for exam in slot:
                    if exam:
                        faculty_available.append(exam.invigilator)
        faculty_count = len(set(faculty_available))
        if faculty_count >= len(self.teachers_data) / 1.8:
            fitness += 5
        else:
            print(
                "[ERR: SFT_CNSTRT 4/4] Grant two hours of break in the week for faculty. Ensure half of the faculty is available at all times for invigilation."
            )

        print(f"\n[CLC_FTNS] Soft Constraint Fitness Sum: {fitness}\n")
        print(f"\n[CLC_FTNS] Final Fitness: {fitness}\n")
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
        mutationPoint2 = random.randint(0, len(child[mutationPoint1]) - 1)
        mutationPoint3 = random.randint(
            0, len(child[mutationPoint1][mutationPoint2]) - 1
        )
        mutationPoint4 = random.randint(
            0, len(child[mutationPoint1][mutationPoint2]) - 1
        )

        (
            child[mutationPoint1][mutationPoint2][mutationPoint3],
            child[mutationPoint1][mutationPoint2][mutationPoint4],
        ) = (
            child[mutationPoint1][mutationPoint2][mutationPoint4],
            child[mutationPoint1][mutationPoint2][mutationPoint3],
        )

        return child

    def genetic_algorithm(self):
        # Initialize population
        population = self.initialize_population()

        # Evolution loop
        for generation in range(self.max_iterations):
            print(f"Generation: {generation + 1}")
            # Calculate fitness for each individual in the population
            valid_schedules = []
            for i, schedule in enumerate(population):
                print(f"[GEN_ALGO]Schedule {i + 1}:")
                fitness_score = self.calculate_fitness(schedule)
                if fitness_score == 60:  # Total sum of hard constraints
                    valid_schedules.append(schedule)

            # Check if any valid schedule is found
            if valid_schedules:
                print("Valid schedules found:")
                for valid_schedule in valid_schedules:
                    self.print_schedule(valid_schedule)
                break

            # If no valid schedule found, continue with next generation
            print("[GEN_ALGO] No valid schedules found in this generation.")

            # Select parents for crossover using roulette wheel selection
            selected_parents = []
            fitness_scores = [self.calculate_fitness(schedule) for schedule in population]
            total_fitness = sum(fitness_scores)
            probabilities = [score / total_fitness for score in fitness_scores]
            for _ in range(len(population)):
                selected_parents.append(
                    random.choices(population, weights=probabilities)[0]
                )

            # Perform crossover to produce children
            children = []
            for i in range(0, len(selected_parents), 2):
                child1, child2 = self.one_point_crossover(
                    selected_parents[i], selected_parents[i + 1]
                )
                # Apply mutation to children
                if np.random.rand() < self.mutation_rate:
                    child1 = self.swapMutation(child1)
                if np.random.rand() < self.mutation_rate:
                    child2 = self.swapMutation(child2)
                children.extend([child1, child2])

            # Replace the current population with the children
            population = children

        else:
            print(
                f"[GEN_ALGO] Termination condition not met after {self.max_iterations} iterations."
            )

        print("[GEN_ALGO] Genetic Algorithm completed.")
        # Print the days the schedule follows
        print(f"[GEN_ALGO] Days of the exams: {self.days}\n")
        print("[GEN_ALGO] Final Schedule:")
        self.print_schedule()


if __name__ == "__main__":
    try:
        # Generate Classrooms C301 thru C315 and C401 thru C415
        classroomsAvailable = []
        for i in range(301, 316):
            classroomsAvailable.append(f"C{i}")
        for i in range(401, 416):
            classroomsAvailable.append(f"C{i}")
        schedule = Schedule(
            population_size=10,
            mutation_rate=0.5,
            max_iterations=100,
            days=3,
            slotsPerDay=3,
            examsPerSlot=5,
            examDuration=2,
            classrooms=classroomsAvailable,
        )
        schedule.initialize_schedule()
        schedule.genetic_algorithm()
    except KeyboardInterrupt:
        print("[MAIN] Program terminated by user.")