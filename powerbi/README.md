# Power BI Dashboard — Student Performance Analysis

This folder contains the Power BI-ready data package and screenshots for the Student Performance Analysis dashboard.

## Dashboard visuals

- Total Students
- Average Math Score
- Average Reading Score
- Average Writing Score
- Pass/Fail Analysis
- Performance Category
- Test Preparation Course
- Average Math Score by Gender
- Average Reading Score by Gender
- Average Writing Score by Gender
- Average Math Score by Parental Education
- Average Reading Score by Parental Education
- Average Writing Score by Parental Education
- Average Math Score by Race/Ethnicity
- Average Math Score by Lunch

## Power BI calculated fields

### Result
```DAX
Result =
IF(
    (StudentsPerformance[math score] +
    StudentsPerformance[reading score] +
    StudentsPerformance[writing score]) / 3 >= 40,
    "Pass",
    "Fail"
)
```

### Performance Category
```DAX
Performance Category =
VAR AvgScore =
    (
        StudentsPerformance[math score] +
        StudentsPerformance[reading score] +
        StudentsPerformance[writing score]
    ) / 3
RETURN
    SWITCH(
        TRUE(),
        AvgScore >= 80, "Excellent",
        AvgScore >= 60, "Good",
        AvgScore >= 40, "Average",
        "Poor"
    )
```

## Opening the dashboard

The actual `.pbix` file must be opened in **Power BI Desktop**. VS Code can store this folder, the dataset, documentation, and screenshots, but it does not natively open `.pbix` files.
