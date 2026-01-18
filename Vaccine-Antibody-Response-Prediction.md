Vaccine-Antibody-Response-Prediction
================
Fang Wang
2025-03-25

``` r
# LOAD LIBRARIES
library(tidyverse)
library(dplyr)
library(ggplot2)
library(pdp)
library(gt)
library(gtsummary)
library(ggpubr)
library (caret)
library(mgcv)
library(earth)
library(tidyverse)
library(rsample)

# KNIT SETTINGS
knitr::opts_chunk$set(
  message = FALSE,
  warning = FALSE,
  out.width = "90%",
  fig.align = "center"
)

# FIGURE SETTINGS
theme_set(
  theme(
    #legend.position = "bottom",
    plot.title = 
      element_text(hjust = 0.5, 
                              margin = margin(b = 5), 
                              face = "bold"),
    plot.subtitle = 
      element_text(hjust = 0.5, 
                                 margin = margin(b = 10),
                                 color = "azure4", 
                                 face = "bold", size = 8)
  )
)
```

# 1. Background & Objective

This project develops and validates a predictive model for
log-transformed post-vaccination antibody levels using demographic and
clinical variables. We benchmark several model families and select a
final model balancing predictive performance and interpretability, then
evaluate generalizability on an independent external cohort.

# 2. Data Source & Cohort Definition

Two de-identified datasets with identical schema are provided: • dat1:
development cohort (used for training + internal test) • dat2:
independent cohort for external validation

Outcome: log_antibody Key predictors include demographics (age, sex,
race/ethnicity), comorbidities (diabetes, hypertension), clinical
measures (BMI, SBP, LDL), and time since vaccination (time).

``` r
# =========================
# 2) Load Data
# =========================

load("data/dat1.RData")
load("data/dat2.RData")

dat1 = as_tibble(dat1) |> 
  janitor::clean_names() |> 
  mutate(across(c(gender, race, smoking, hypertension, diabetes), as.factor))

dat2 = as_tibble(dat2) |> 
  janitor::clean_names() |> 
  mutate(across(c(gender, race, smoking, hypertension, diabetes), as.factor))

dat1_decoded = dat1 |> 
  mutate(
    across(c(gender, race, smoking, hypertension, diabetes), as.character),
    across(c(gender, race, smoking, hypertension, diabetes), as.numeric),
    gender = factor(
      case_match(gender, 0 ~ "Female", 1 ~ "Male"), 
      levels = c("Female", "Male")),
    race = factor(
      case_match(race, 1 ~ "White", 2 ~ "Asian", 3 ~ "Black", 4 ~ "Hispanic"), 
      levels = c("White", "Asian", "Black", "Hispanic")),
    smoking = factor(
      case_match(smoking, 
                 0 ~ "Never smoked", 
                 1 ~ "Former smoker", 
                 2 ~ "Current smoker"), 
      levels = c("Never smoked", "Former smoker", "Current smoker")),
    hypertension = factor(
      case_match(hypertension, 0 ~ "No", 1 ~ "Yes"), 
      levels = c("No", "Yes")),
    diabetes = factor(
      case_match(diabetes, 0 ~ "No", 1 ~ "Yes"), 
      levels = c("No", "Yes"))
  )

head(dat1_decoded, 5) |> 
  knitr::kable(digits = 2)
```

| id | age | gender | race | smoking | height | weight | bmi | diabetes | hypertension | sbp | ldl | time | log_antibody |
|---:|---:|:---|:---|:---|---:|---:|---:|:---|:---|---:|---:|---:|---:|
| 1 | 50 | Female | White | Never smoked | 176.1 | 68.3 | 22.0 | No | No | 130 | 82 | 76 | 10.65 |
| 2 | 71 | Male | White | Never smoked | 175.7 | 69.6 | 22.6 | No | Yes | 149 | 129 | 82 | 9.89 |
| 3 | 58 | Male | White | Former smoker | 168.7 | 76.9 | 27.0 | No | No | 127 | 101 | 168 | 10.90 |
| 4 | 63 | Female | White | Never smoked | 167.4 | 90.0 | 32.1 | No | Yes | 138 | 93 | 105 | 9.91 |
| 5 | 56 | Male | White | Never smoked | 162.7 | 83.9 | 31.7 | No | No | 123 | 97 | 193 | 9.56 |

# 3. Cohort Characteristics & Clinically Relevant EDA

3.1 Cohort characteristics (dat1)

``` r
dat1_decoded |> 
  select(-id) |> 
  tbl_summary(
    statistic = list (all_continuous() ~ "{min} - {max} | Mean: {mean} (SD: {sd})", 
                      all_categorical() ~ "{n} ({p}%)"),
    label = list(
      id = "ID (id)",
      age = "Age (age) ",
      gender = "Gender (gender)",
      race = "Race/ethnicity (race)",
      smoking = "Smoking (smoking)",
      height = "Height (height)",
      weight = "Weight (weight)",
      bmi = "BMI (bmi)",
      diabetes = "Diabetes (diabetes)",
      hypertension = "Hypertension (hypertension)",
      sbp = "Systolic blood pressure (sbp)",
      ldl = "LDL cholesterol (ldl)",
      time = "Time since vaccination (time)",
      log_antibody = "Log-transformed antibody level (log_antibody)"
    )
  ) |> 
  modify_caption("**Summary Statistics for Dataset 1**")
```

<div id="rjzbmqetwb" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#rjzbmqetwb table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
&#10;#rjzbmqetwb thead, #rjzbmqetwb tbody, #rjzbmqetwb tfoot, #rjzbmqetwb tr, #rjzbmqetwb td, #rjzbmqetwb th {
  border-style: none;
}
&#10;#rjzbmqetwb p {
  margin: 0;
  padding: 0;
}
&#10;#rjzbmqetwb .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;#rjzbmqetwb .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}
&#10;#rjzbmqetwb .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}
&#10;#rjzbmqetwb .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}
&#10;#rjzbmqetwb .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}
&#10;#rjzbmqetwb .gt_column_spanner_outer:first-child {
  padding-left: 0;
}
&#10;#rjzbmqetwb .gt_column_spanner_outer:last-child {
  padding-right: 0;
}
&#10;#rjzbmqetwb .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}
&#10;#rjzbmqetwb .gt_spanner_row {
  border-bottom-style: hidden;
}
&#10;#rjzbmqetwb .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}
&#10;#rjzbmqetwb .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}
&#10;#rjzbmqetwb .gt_from_md > :first-child {
  margin-top: 0;
}
&#10;#rjzbmqetwb .gt_from_md > :last-child {
  margin-bottom: 0;
}
&#10;#rjzbmqetwb .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}
&#10;#rjzbmqetwb .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#rjzbmqetwb .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}
&#10;#rjzbmqetwb .gt_row_group_first td {
  border-top-width: 2px;
}
&#10;#rjzbmqetwb .gt_row_group_first th {
  border-top-width: 2px;
}
&#10;#rjzbmqetwb .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#rjzbmqetwb .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_first_summary_row.thick {
  border-top-width: 2px;
}
&#10;#rjzbmqetwb .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#rjzbmqetwb .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}
&#10;#rjzbmqetwb .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#rjzbmqetwb .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}
&#10;#rjzbmqetwb .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#rjzbmqetwb .gt_left {
  text-align: left;
}
&#10;#rjzbmqetwb .gt_center {
  text-align: center;
}
&#10;#rjzbmqetwb .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
&#10;#rjzbmqetwb .gt_font_normal {
  font-weight: normal;
}
&#10;#rjzbmqetwb .gt_font_bold {
  font-weight: bold;
}
&#10;#rjzbmqetwb .gt_font_italic {
  font-style: italic;
}
&#10;#rjzbmqetwb .gt_super {
  font-size: 65%;
}
&#10;#rjzbmqetwb .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}
&#10;#rjzbmqetwb .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}
&#10;#rjzbmqetwb .gt_indent_1 {
  text-indent: 5px;
}
&#10;#rjzbmqetwb .gt_indent_2 {
  text-indent: 10px;
}
&#10;#rjzbmqetwb .gt_indent_3 {
  text-indent: 15px;
}
&#10;#rjzbmqetwb .gt_indent_4 {
  text-indent: 20px;
}
&#10;#rjzbmqetwb .gt_indent_5 {
  text-indent: 25px;
}
&#10;#rjzbmqetwb .katex-display {
  display: inline-flex !important;
  margin-bottom: 0.75em !important;
}
&#10;#rjzbmqetwb div.Reactable > div.rt-table > div.rt-thead > div.rt-tr.rt-tr-group-header > div.rt-th-group:after {
  height: 0px !important;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <caption><span class='gt_from_md'><strong>Summary Statistics for Dataset 1</strong></span></caption>
  <thead>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="label"><span class='gt_from_md'><strong>Characteristic</strong></span></th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="stat_0"><span class='gt_from_md'><strong>N = 5,000</strong></span><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;"><sup>1</sup></span></th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers="label" class="gt_row gt_left">Age (age) </td>
<td headers="stat_0" class="gt_row gt_center">44.0 - 75.0 | Mean: 60.0 (SD: 4.5)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Gender (gender)</td>
<td headers="stat_0" class="gt_row gt_center"><br /></td></tr>
    <tr><td headers="label" class="gt_row gt_left">    Female</td>
<td headers="stat_0" class="gt_row gt_center">2,573 (51%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">    Male</td>
<td headers="stat_0" class="gt_row gt_center">2,427 (49%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Race/ethnicity (race)</td>
<td headers="stat_0" class="gt_row gt_center"><br /></td></tr>
    <tr><td headers="label" class="gt_row gt_left">    White</td>
<td headers="stat_0" class="gt_row gt_center">3,221 (64%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">    Asian</td>
<td headers="stat_0" class="gt_row gt_center">278 (5.6%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">    Black</td>
<td headers="stat_0" class="gt_row gt_center">1,036 (21%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">    Hispanic</td>
<td headers="stat_0" class="gt_row gt_center">465 (9.3%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Smoking (smoking)</td>
<td headers="stat_0" class="gt_row gt_center"><br /></td></tr>
    <tr><td headers="label" class="gt_row gt_left">    Never smoked</td>
<td headers="stat_0" class="gt_row gt_center">3,010 (60%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">    Former smoker</td>
<td headers="stat_0" class="gt_row gt_center">1,504 (30%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">    Current smoker</td>
<td headers="stat_0" class="gt_row gt_center">486 (9.7%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Height (height)</td>
<td headers="stat_0" class="gt_row gt_center">150.2 - 192.9 | Mean: 170.1 (SD: 5.9)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Weight (weight)</td>
<td headers="stat_0" class="gt_row gt_center">57 - 106 | Mean: 80 (SD: 7)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">BMI (bmi)</td>
<td headers="stat_0" class="gt_row gt_center">18.20 - 38.80 | Mean: 27.74 (SD: 2.76)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Diabetes (diabetes)</td>
<td headers="stat_0" class="gt_row gt_center">772 (15%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Hypertension (hypertension)</td>
<td headers="stat_0" class="gt_row gt_center">2,298 (46%)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Systolic blood pressure (sbp)</td>
<td headers="stat_0" class="gt_row gt_center">101 - 155 | Mean: 130 (SD: 8)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">LDL cholesterol (ldl)</td>
<td headers="stat_0" class="gt_row gt_center">43 - 185 | Mean: 110 (SD: 20)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Time since vaccination (time)</td>
<td headers="stat_0" class="gt_row gt_center">30 - 270 | Mean: 109 (SD: 43)</td></tr>
    <tr><td headers="label" class="gt_row gt_left">Log-transformed antibody level (log_antibody)</td>
<td headers="stat_0" class="gt_row gt_center">7.77 - 11.96 | Mean: 10.06 (SD: 0.60)</td></tr>
  </tbody>
  &#10;  <tfoot class="gt_footnotes">
    <tr>
      <td class="gt_footnote" colspan="2"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;"><sup>1</sup></span> <span class='gt_from_md'>Min - Max | Mean: Mean (SD: SD); n (%)</span></td>
    </tr>
  </tfoot>
</table>
</div>

3.2 EDA (dat1)

``` r
## Categorical Variable Barplots

p_gender = ggplot(dat1_decoded, aes(x = gender)) + 
  geom_bar(fill = "skyblue", color = "black", lwd = 0.5) + 
  labs(title = "Gender\nDistribution", x = "Gender", y = "Count")

p_race = ggplot(dat1_decoded, aes(x = race)) + 
  geom_bar(fill= "orange", color = "black", lwd = 0.5) + 
  labs(title = "Race/Ethnicity\nDistribution", 
       x = "Race/Ethnicity", 
       y = "Count")

p_smoking = dat1_decoded |> 
  mutate(smoking = fct_rev(as.factor(str_replace(smoking, " ", "\n")))) |> 
  ggplot(aes(x = smoking)) + 
  geom_bar(fill = "skyblue", color = "black", lwd = 0.5) + 
  labs(title = "Smoking Status\nDistribution", 
       x = "Smoking", 
       y = "Count")

p_diabetes = ggplot(dat1_decoded, aes(x = diabetes)) + 
  geom_bar(fill = "orange", color = "black", lwd = 0.5) + 
  labs(title = "Diabetes Status\nDistribution", 
       x = "Diabetes", 
       y = "Count")

p_hypertension = ggplot(dat1_decoded, aes(x = hypertension)) + 
  geom_bar(fill = "skyblue", color = "black", lwd = 0.5) + 
  labs(title = "Hypertension Status\nDistribution", 
       x = "Hypertension", 
       y = "Count")

barplot_combined = ggarrange(p_gender, 
                             p_race, 
                             p_smoking, 
                             p_diabetes, 
                             p_hypertension, 
                             nrow = 2, 
                             ncol = 3)

barplot_combined
```

<img src="Vaccine-Antibody-Response-Prediction_files/figure-gfm/unnamed-chunk-3-1.png" width="90%" style="display: block; margin: auto;" />

``` r
## Numeric Variable Histograms

p_age = ggplot(dat1_decoded, aes(x = age)) + 
  geom_histogram(fill = "skyblue", color = "black", lwd = 0.5) + 
  labs(title = "Age\nDistribution", 
       x = "Age", 
       y = "Count")

p_height = ggplot(dat1_decoded, aes(x = height)) + 
  geom_histogram(fill = "orange", color = "black", lwd = 0.5) + 
  labs(title = "Height\nDistribution", 
       x = "Height", 
       y = "Count")

p_weight = ggplot(dat1_decoded, aes(x = weight)) + 
  geom_histogram(fill = "skyblue", color = "black", lwd = 0.5) + 
  labs(title = "Weight\nDistribution", 
       x = "Weight", 
       y = "Count")

p_bmi = ggplot(dat1_decoded, aes(x = bmi)) + 
  geom_histogram(fill = "orange", color = "black", lwd = 0.5) + 
  labs(title = "Body Mass Index\nDistribution", 
       x = "BMI", 
       y = "Count")

p_sbp = ggplot(dat1_decoded, aes(x = sbp)) + 
  geom_histogram(fill = "skyblue", color = "black", lwd = 0.5) + 
  labs(title = "Systolic Blood Pressure\nDistribution", 
       x = "Systolic Blood Pressure", 
       y = "Count")

p_ldl = ggplot(dat1_decoded, aes(x = ldl)) + 
  geom_histogram(fill = "orange", color = "black", lwd = 0.5) + 
  labs(title = "LDL Cholesterol\nDistribution", 
       x = "LDL Cholesterol", 
       y = "Count")

p_time = ggplot(dat1_decoded, aes(x = time)) + 
  geom_histogram(fill = "skyblue", color = "black", lwd = 0.5) + 
  labs(title = "Time Since Vax\nDistribution", 
       x = "Time Since Vaccination", 
       y = "Count")

histogram_combined = ggarrange(p_age, 
                               p_height, 
                               p_weight, 
                               p_bmi, 
                               p_sbp, 
                               p_ldl, 
                               p_time, 
                               nrow = 3, 
                               ncol = 3)

histogram_combined
```

<img src="Vaccine-Antibody-Response-Prediction_files/figure-gfm/unnamed-chunk-4-1.png" width="90%" style="display: block; margin: auto;" />

# 4. Model Development Preprocessing

• Split dat1 into training (80%) and internal test (20%) using a fixed
seed. • Bivariate Plots before model training

``` r
# =========================
# 4) Train/Test Split + CV control
# =========================

# split dataset into training data and testing data:
set.seed(2025)
data_split = initial_split(dat1, prop =0.8)
training_data = training (data_split)
testing_data = testing (data_split)
# Cross-validation setup:
ctrl1 = trainControl(method = "cv", number = 10)
```

``` r
# =========================
# 4.1 Bivariate Plots: Density Plots of Response Variable by Categorical Variables
# =========================

# Create function for plot generation
cat_plot = function(var, label, df = dat1_decoded){
  output_plot = df |> 
    ggplot(aes(x = log_antibody, fill = !!sym(var))) +
    geom_density(alpha = 0.5, lwd = 0.5) +
    labs(
      title = paste0("Log Antibody vs. ", label),
      x = "Log Antibody",
      y = "Density",
      fill = label
    )
  
  return(output_plot)
}

training_data_decoded = dat1_decoded |> 
  filter(id %in% training_data$id)

# Generate Plots
gender_plot = cat_plot("gender", 
                       "Gender", 
                       training_data_decoded)
race_plot = cat_plot("race", 
                     "Race", 
                     training_data_decoded)
smoking_plot = cat_plot("smoking", 
                        "Smoking Status", 
                        training_data_decoded)
hypertension_plot = cat_plot("hypertension", 
                             "Hypertension", 
                             training_data_decoded)
diabetes_plot = cat_plot("diabetes", 
                         "Diabetes", 
                         training_data_decoded)

# Display multi-plot figure (manually exporting to save multi-plot figure)
cat_vars_plot = ggarrange(gender_plot, 
                          race_plot, 
                          smoking_plot, 
                          hypertension_plot, 
                          diabetes_plot, 
                          nrow = 3, 
                          ncol = 2)
cat_vars_plot
```

<img src="Vaccine-Antibody-Response-Prediction_files/figure-gfm/unnamed-chunk-5-1.png" width="90%" style="display: block; margin: auto;" />

``` r
# =========================
# 4.2 Bivariate Plots: Scatterplots for Response with Numeric Variables
# =========================

# Create function for plot generation
num_plot = function(var, label, xlab, df = dat1){
  output_plot = df |> 
    ggplot(aes(x = !!sym(var), y = log_antibody)) +
    geom_point(size = 0.3) +
    geom_smooth(lwd = 0.5, method = "loess", color = "red") +
    labs(
      title = paste0("Log Antibody vs. ", label),
      x = xlab,
      y = "Log Antibody"
    )
  
  return(output_plot)
}

# Generate Plots
age_plot = num_plot("age", "Age", "Age (years)", training_data)
height_plot = num_plot("height", "Height", "Height", training_data)
weight_plot = num_plot("weight", "Weight", "Weight", training_data)
bmi_plot = num_plot("bmi", "BMI", "Body Mass Index", training_data)
sbp_plot = num_plot("sbp", "SBP", "Systolic Blood Pressure", training_data)
ldl_plot = num_plot("ldl", "LDL", "LDL Cholesterol", training_data)
time_plot = num_plot("time", "Time", "Time Since Vaccination", training_data)

# Display multi-plot figure (manually exporting to save multi-plot figure)
cont_vars_plot = ggarrange(age_plot, 
                           height_plot, 
                           weight_plot, 
                           bmi_plot, 
                           sbp_plot, 
                           ldl_plot, 
                           time_plot, 
                           nrow = 3, 
                           ncol= 3)
cont_vars_plot
```

<img src="Vaccine-Antibody-Response-Prediction_files/figure-gfm/unnamed-chunk-6-1.png" width="90%" style="display: block; margin: auto;" />

``` r
# =========================
# 4.3 Correlation Plot
# =========================
dat1_x_matrix = model.matrix(log_antibody ~ ., data = dat1)[, -1]
corrplot::corrplot(cor(dat1_x_matrix), method = "circle", type = "full")
```

<img src="Vaccine-Antibody-Response-Prediction_files/figure-gfm/unnamed-chunk-7-1.png" width="90%" style="display: block; margin: auto;" />
The correlation matrix indicates that the majority of predictors are
only weakly correlated. Strong positive correlations are observed
between weight and BMI, and moderate correlations between height and
weight, reflecting expected anthropometric relationships. Among
cardiometabolic variables, systolic blood pressure (SBP) is positively
correlated with hypertension status, and modest correlations are also
seen between SBP and LDL cholesterol.

Importantly, no pair of predictors demonstrates extremely high
correlation, suggesting that multicollinearity is unlikely to
substantially bias model estimation.

# 5. Model Development

• Benchmark model families with 10-fold cross-validation: \#Linear
regression (OLS) \#Elastic Net (glmnet) \#Partial Least Squares (PLS)
\#Generalized Additive Model (GAM) MARS (earth) • Primary metric: RMSE;
secondary: MAE and R².

5.1 Create model matrix (for algorithms that require x/y)

``` r
# training_data
x = model.matrix(log_antibody~ .- id, training_data)[,-1]
y = training_data$log_antibody
```

5.2 Candidate models

``` r
set.seed(2025)

# 1) OLS 
lm_fit = train(
  x = x, 
  y = y,
  method = "lm",
  trControl = ctrl1
)

# 2) Elastic Net
enet_fit = train(
    x = x,
    y = y,
    method = "glmnet",
    tuneGrid = expand.grid(
        alpha = seq(0, 1, length = 21),
        lambda = exp(seq(9, -3, length = 100))
    ),
    trControl = ctrl1
)


# 3) PLS
pls_fit = train(x, y,
  method = "pls",
  tuneGrid = data.frame(ncomp = 1:14),
  trControl = ctrl1,
  preProcess = c("center", "scale"))

# 4) GAM (formula-based for interpretability)
gam_fit = train(
  log_antibody ~ age + height + weight + bmi + sbp + ldl + time + gender + race + smoking + diabetes + hypertension,
  data = training_data,
  method = "gam",
  trControl = ctrl1
)

# 5) MARS

mars_grid = expand.grid(degree = 1:4, nprune = 2:20)
mars_fit = train(x, y,
                  method = "earth",
                  tuneGrid = mars_grid,
                  trControl = ctrl1)
```

5.3 Benchmark results (CV)

``` r
resamp = resamples(list(OLS = lm_fit,
                        ENET = enet_fit,
                        PLS = pls_fit,
                        GAM = gam_fit, 
                        MARS = mars_fit))
summary(resamp)
```

    ## 
    ## Call:
    ## summary.resamples(object = resamp)
    ## 
    ## Models: OLS, ENET, PLS, GAM, MARS 
    ## Number of resamples: 10 
    ## 
    ## MAE 
    ##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## OLS  0.4283438 0.4376430 0.4438661 0.4445266 0.4503930 0.4652542    0
    ## ENET 0.4287823 0.4403623 0.4468050 0.4451614 0.4517333 0.4590639    0
    ## PLS  0.4212553 0.4366558 0.4457407 0.4440142 0.4552666 0.4632166    0
    ## GAM  0.4007450 0.4135072 0.4316845 0.4271420 0.4399630 0.4446842    0
    ## MARS 0.4023337 0.4185781 0.4267958 0.4266424 0.4328682 0.4597035    0
    ## 
    ## RMSE 
    ##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## OLS  0.5386739 0.5453977 0.5557110 0.5560978 0.5632708 0.5835946    0
    ## ENET 0.5426355 0.5523170 0.5576403 0.5574020 0.5618173 0.5714932    0
    ## PLS  0.5218279 0.5466547 0.5555585 0.5557040 0.5701818 0.5757101    0
    ## GAM  0.5127879 0.5198475 0.5368367 0.5325227 0.5450922 0.5482469    0
    ## MARS 0.5041308 0.5235546 0.5287436 0.5320812 0.5340485 0.5801572    0
    ## 
    ## Rsquared 
    ##            Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## OLS  0.08736256 0.1178073 0.1330346 0.1382936 0.1601209 0.2113563    0
    ## ENET 0.10353164 0.1127225 0.1250057 0.1340032 0.1578014 0.1819220    0
    ## PLS  0.08360088 0.1060300 0.1460962 0.1397179 0.1635200 0.2192020    0
    ## GAM  0.14197736 0.1878285 0.2066508 0.2097358 0.2406073 0.2644246    0
    ## MARS 0.15886721 0.1929210 0.2127702 0.2117636 0.2282614 0.2716848    0

``` r
bwplot(resamp, metric = "RMSE")
```

<img src="Vaccine-Antibody-Response-Prediction_files/figure-gfm/unnamed-chunk-9-1.png" width="90%" style="display: block; margin: auto;" />

``` r
png("figures/model_comparison.png", width = 1400, height = 800, res = 200)
```

# 6. Final Model Selection & Interpretability

We select the model with best cross-validated RMSE while maintaining
interpretability. In this analysis, GAM provides strong performance and
interpretable non-linear effects.

``` r
plot(gam_fit$finalModel, pages = 1, shade = TRUE, seWithMean = TRUE)
```

<img src="Vaccine-Antibody-Response-Prediction_files/figure-gfm/unnamed-chunk-10-1.png" width="90%" style="display: block; margin: auto;" />

``` r
# If you want to save this plot:

png("figures/gam_smooth_terms.png", width = 1400, height = 800, res = 200)
plot(gam_fit$finalModel, pages = 1, shade = TRUE, seWithMean = TRUE)
dev.off()
```

    ## quartz_off_screen 
    ##                 2

# 7. Internal Test Performance (dat1 hold-out)

``` r
final_model <- gam_fit
pred_test <- predict(final_model, newdata = testing_data)

rmse_internal <- sqrt(mean((pred_test - testing_data$log_antibody)^2))
mae_internal  <- mean(abs(pred_test - testing_data$log_antibody))

rmse_internal
```

    ## [1] 0.5121094

``` r
mae_internal
```

    ## [1] 0.4053604

``` r
# Pred vs Observed plot (internal)

p_int <- tibble(
observed = testing_data$log_antibody,
predicted = pred_test
) |>
ggplot(aes(x = observed, y = predicted)) +
geom_point(alpha = 0.35, size = 0.7) +
geom_abline(slope = 1, intercept = 0, lwd = 0.7) +
labs(
title = "Internal Test: Predicted vs Observed (dat1 test)",
x = "Observed log antibody",
y = "Predicted log antibody"
)

p_int
```

<img src="Vaccine-Antibody-Response-Prediction_files/figure-gfm/unnamed-chunk-11-1.png" width="90%" style="display: block; margin: auto;" />

``` r
ggsave("figures/pred_vs_obs_internal.png", p_int, width = 6.5, height = 4, dpi = 300)
```

The model shows good agreement between predicted and observed antibody
levels in the internal test set, with most predictions closely aligned
to the 45-degree reference line.

# 8. External Validation (independent cohort: dat2)

``` r
pred_ext <- predict(final_model, newdata = dat2)

rmse_external <- sqrt(mean((pred_ext - dat2$log_antibody)^2))
mae_external  <- mean(abs(pred_ext - dat2$log_antibody))

rmse_external
```

    ## [1] 0.543914

``` r
mae_external
```

    ## [1] 0.4364023

``` r
# Pred vs Observed plot (external)

p_ext <- tibble(
observed = dat2$log_antibody,
predicted = pred_ext
) |>
ggplot(aes(x = observed, y = predicted)) +
geom_point(alpha = 0.35, size = 0.7) +
geom_abline(slope = 1, intercept = 0, lwd = 0.7) +
labs(
title = "External Validation: Predicted vs Observed (dat2)",
x = "Observed log antibody",
y = "Predicted log antibody"
)

p_ext
```

<img src="Vaccine-Antibody-Response-Prediction_files/figure-gfm/unnamed-chunk-12-1.png" width="90%" style="display: block; margin: auto;" />

``` r
ggsave("figures/pred_vs_obs_external.png", p_ext, width = 6.5, height = 4, dpi = 300)
```

While prediction accuracy decreases modestly in the external cohort, the
model retains a clear linear association between predicted and observed
values, indicating reasonable transportability.

# 9. Conclusion

Predicted-versus-observed plots demonstrate good calibration in the
internal test set, with predictions closely aligned to the identity
line. In external validation, prediction uncertainty increases and mild
underestimation is observed at higher antibody levels, consistent with
expected cohort shift. Importantly, the overall linear association is
preserved, indicating reasonable generalizability of the model beyond
the development cohort.

# 10. Limitations & Next Steps

•This is an observational modeling exercise; associations do not imply
causation. •Potential cohort shift between dat1 and dat2 may affect
transportability. •Next steps could include calibration assessment,
feature engineering, and sensitivity analyses.
