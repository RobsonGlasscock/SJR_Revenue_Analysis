# %%
%reset -f
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.dates as mdates

"""
############  Attributions ###################
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/date.html
# https://stackoverflow.com/questions/9750330/how-to-convert-integer-into-date-object-python/37674465
# https://stackoverflow.com/questions/2623156/how-to-convert-the-integer-date-format-into-yyyymmdd
# https://stackoverflow.com/questions/40511476/how-to-properly-use-funcformatterfunc
# https://stackoverflow.com/questions/58881360/python-plot-shows-numbers-instead-of-dates-on-x-axis

# Yves Hilpisch "Python for Finance"
# Theodore Petrou "Pandas Cookbook"
# Joel Grus "Data Science from Scratch"
# Daniel Chen "Pandas for Everyone"
# Wes McKinney "Python for Data Analysis"
# Jake VanderPlas "Python Data Science Handbook"
#############################################
"""
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:,}".format

# %%
df = pd.read_excel("sjr.xlsx")
df = df.iloc[:16]
df.head()

# %%
df.info()

# %%
# Segment revenue graph- object oriented approach.
fig, ax = plt.subplots()
# Create first time series line with the appropriate label.
ax.scatter(df["date"], df["wline_rev"], label="Wireline")
# Create second time series line with the appropriate lable.
ax.scatter(df["date"], df["wless_rev"], label="Wireless")
# Title of graph.
ax.set_title("Segment Revenue")
# Labels for the x and y axes, respectively.
ax.set_xlabel("Quarter")
ax.set_ylabel("CAD (Millions)")
# Feed in the ticks for the x axis, rotate them, and display as dates.
ax.set_xticklabels(ax.get_xticks(), rotation="vertical")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
# Specify the maximum number of ticks. This option makes the ticks
# line up better and run through all of the time series. Without
# this option the last time tick is 2020-02.
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
# Include the legend in the graph.
ax.legend()
# Format the y axis numbers to have a comma separater.
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
# Save the figure and the "bbox_inches" option keeps the saved
# image from having the x axis labels cut off.
plt.savefig("Segment_Revenue.pdf", bbox_inches="tight")

# %%
# Total Revenue graph- object oriented approach.
fig, ax = plt.subplots()
# Create first time series line with the appropriate label.
ax.scatter(df["date"], df["tot_rev_mils"])
# Title of graph.
ax.set_title("Total Revenue")
# Labels for the x and y axes, respectively.
ax.set_xlabel("Quarter")
ax.set_ylabel("CAD (Millions)")
# Feed in the ticks for the x axis, rotate them, and display as dates.
ax.set_xticklabels(ax.get_xticks(), rotation="vertical")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
# Specify the maximum number of ticks. This option makes the ticks
# line up better and run through all of the time series. Without
# this option the last time tick is 2020-02.
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
# Format the y axis numbers to have a comma separater.
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
# Save the figure and the "bbox_inches" option keeps the saved
# image from having the x axis labels cut off.
plt.savefig("Total_Revenue.pdf", bbox_inches="tight")

# %%
# Individual Segment Models #
# Estimate Wireline average revenue per customer 2/28/17 through 11/30/19.
# This is the training period. The "test set" for "out of sample"
# model accuracy tests is 2/29/20 through 11/30/20.
y = df["wline_rev"].iloc[4:]
x = df["wline_cust_tot"].iloc[4:]
model = sm.OLS(y, x).fit()
model.summary()
model.params
wline_mfx = model.params[0] * 1000000
wline_mfx

# %%
# Individual Segment Model
# Estimate Wireless average revenue per customer for the same training period.
y = df["wless_rev"].iloc[4:]
x = df["wless_cust_tot"].iloc[4:]
model = sm.OLS(y, x).fit()
model.summary()
wless_mfx = model.params[0] * 1000000
wless_mfx

# %%
print(
    "The estimated difference in revenue per customer from\
    the segment models is",
    wline_mfx - wless_mfx,
)

# %%
# Total Revenue Model
# Estimate Wireline average revenue per customer 2/28/17 through 11/30/19.
# This is the training period. The "test set" period for "out of sample"
# model accuracy tests is 2/29/20 through 11/30/20.
y = df["tot_rev_mils"].iloc[4:]
x = df[["wline_cust_tot", "wless_cust_tot"]].iloc[4:]
model = sm.OLS(y, x).fit()
model.summary()
model.params
wline_nested_slope = model.params[0] * 1000000
wless_nested_slope = model.params[1] * 1000000

# %%
print(wline_nested_slope)
print(wless_nested_slope)
print(
    "The estimated difference in revenue per customer from\
    the total revenue model is",
    wline_nested_slope - wless_nested_slope,
)

# The individual segment model estimates Wireline at $191 and Wireless at $170.
# The total revenue model estimates Wireline at $176 and Wireless at $232.

# %%
# Averaging Approach
# Alternative approach to average revenue per customer using averages
df["wline_avg_per_cust"] = (
    df["wline_rev"].iloc[4:] / df["wline_cust_tot"].iloc[4:]
) * 1000000
df["wless_avg_per_cust"] = (
    df["wless_rev"].iloc[4:] / df["wless_cust_tot"].iloc[4:]
) * 1000000

# %%
print(df["wline_avg_per_cust"].mean())
print(df["wless_avg_per_cust"].mean())

print(
    "The estimated differnce in revenue per customer from averaging is",
    df["wline_avg_per_cust"].mean() - df["wless_avg_per_cust"].mean(),
)

# %%
# Comparison of total revenue predictions from 1) Individual Segment Models and
# 2) Total Revenue Model for the "test set" sample period of
# 2/29/20 through 11/30/20.
df["indiv_yhat"] = wline_mfx * df["wline_cust_tot"] \
                   + (wless_mfx * df["wless_cust_tot"])


df["indiv_yhat"] = (df["indiv_yhat"] / 1000000).round()

df["indiv_wline_yhat"] = wline_mfx * df["wline_cust_tot"]
df["indiv_wless_yhat"] = wless_mfx * df["wless_cust_tot"]

df["indiv_wline_yhat"] = (df["indiv_wline_yhat"] / 1000000).round()
df["indiv_wless_yhat"] = (df["indiv_wless_yhat"] / 1000000).round()

df["total_yhat"] = wline_nested_slope * df["wline_cust_tot"] \
                   + (wless_nested_slope * df["wless_cust_tot"])

df["total_yhat"] = (df["total_yhat"] / 1000000).round()

df["total_wline_yhat"] = wline_nested_slope * df["wline_cust_tot"]

df["total_wless_yhat"] = wless_nested_slope * df["wless_cust_tot"]

df["total_wline_yhat"] = (df["total_wline_yhat"] / 1000000).round()
df["total_wless_yhat"] = (df["total_wless_yhat"] / 1000000).round()

df["indiv_sr"] = (df["indiv_yhat"].iloc[:4] - df["tot_rev_mils"].iloc[:4]) ** 2
df["total_sr"] = (df["total_yhat"].iloc[:4] - df["tot_rev_mils"].iloc[:4]) ** 2

# %%
df[["indiv_sr", "total_sr"]].iloc[:4]

# %%
df[["indiv_sr", "total_sr"]].sum()

# %%
df["indiv_sr"].sum() / df["total_sr"].sum()

# The sum of the squared difference between the predictions and
# actual values in the hold out sample period are approximately
# 2.7 times larger for the Individual Segment Models model compared
# to the Total Revenue Model.

# %%
df[
    [
        "date",
        "tot_rev_mils",
        "indiv_yhat",
        "total_yhat",
        "wline_rev",
        "indiv_wline_yhat",
        "total_wline_yhat",
        "wless_rev",
        "indiv_wless_yhat",
        "total_wless_yhat",
        "indiv_sr",
        "total_sr"
    ]
].iloc[:4]

# Above shows that, while the Total Revenue Model estimates the
# Wireless segment revenue to be much higher than it is, the overall
# performance when predicting total revenue from all segments
# superior to Individual Segment Models.

# %%
# Scatterplot of Wireline customers- object oriented approach.
fig, ax = plt.subplots()
# Create first time series line with the appropriate label.
ax.scatter(df["date"], df["wline_cust_tot"])
# Title of graph.
ax.set_title("Wireline Customers")
# Labels for the x and y axes, respectively.
ax.set_xlabel("Quarter")
ax.set_ylabel("Customers")
# Feed in the ticks for the x axis, rotate them, and display as dates.
ax.set_xticklabels(ax.get_xticks(), rotation="vertical")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
# Specify the maximum number of ticks. This option makes the ticks line up
# better and run through all of the time series. Without this option the
# last time tick is 2020-02.
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
# Format the y axis numbers to have a comma separater.
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
# Save the figure and the "bbox_inches" option keeps the saved image
# from having the x axis labels cut off.
plt.savefig("Wireline_Customers.pdf", bbox_inches="tight")

# %%
# Scatterplot of Wireless customers- object oriented approach.
fig, ax = plt.subplots()
# Create first time series line with the appropriate label.
ax.scatter(df["date"], df["wless_cust_tot"])
# Title of graph.
ax.set_title("Wireless Customers")
# Labels for the x and y axes, respectively.
ax.set_xlabel("Quarter")
ax.set_ylabel("Customers")
# Feed in the ticks for the x axis, rotate them, and display as dates.
ax.set_xticklabels(ax.get_xticks(), rotation="vertical")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
# Specify the maximum number of ticks. This option makes the
# ticks line up better and run through all of the time series.
# Without this option the last time tick is 2020-02.
ax.xaxis.set_major_locator(plt.MaxNLocator(12))
# Format the y axis numbers to have a comma separater.
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
# Save the figure and the "bbox_inches" option keeps the saved image from
# having the x axis labels cut off.
plt.savefig("Wireless_Customers.pdf", bbox_inches="tight")

# %%
# Create a linear trend term. Note that the most recent obs is first
# in the dataset. Sort the data with oldest first before creating the trend.

df.sort_values(by=["date"], inplace=True)
df.reset_index(drop=True, inplace=True)

df["trend"] = df.index + 1
df["trend"]

# %%
# Models to forecast the number of customers for each segment.
y = df["wline_cust_tot"]
x = df["trend"]
x_model = sm.add_constant(x)
model = sm.OLS(y, x_model).fit()
model.summary()
model.params
wline_cons = model.params[0]
wline_slope = model.params[1]

# %%
wline_cons

# %%
wline_slope

# %%
y = df["wless_cust_tot"]
x = df["trend"]
x_model = sm.add_constant(x)
model = sm.OLS(y, x_model).fit()
model.summary()
model.params
wless_cons = model.params[0]
wless_slope = model.params[1]

# %%
wless_cons

# %%
wless_slope

# %%
# Create a dataframe with a one year ahead out of sample period, quarterly.
df_newdates = pd.DataFrame(
    ["20210228", "20210531", "20210831", "20211130"], columns=["date"]
)

# Format the dates as datetimes.
df_newdates["date"] = pd.to_datetime(df_newdates["date"], format="%Y%m%d")

# %%
df_newdates

# %%
df_newdates.info()

# %%
# Append the out of sample quarters into the original dataframe.
df = df.append(df_newdates)
df.reset_index(drop=True, inplace=True)
df

# %%
# Recalculate the time trend.
df["trend"] = df.index + 1

# Re-estimate the marginal effects of each Wireline and Wireless customer
# on total revenue using the Total Revenue Model on the entire sample period.
y = df["tot_rev_mils"].iloc[:16]
x = df[["wline_cust_tot", "wless_cust_tot"]].iloc[:16]
model = sm.OLS(y, x).fit()
model.summary()
model.params
wline_nested_slope = model.params[0] * 1000000
wless_nested_slope = model.params[1] * 1000000

# %%
wline_nested_slope

# %%
wless_nested_slope

# %%
# Create one-year ahead predictions for customers by segment from the
# OLS estimates.
df["wline_cust_pred"] = (wline_cons + (wline_slope * df["trend"])).round()
df["wless_cust_pred"] = (wless_cons + (wless_slope * df["trend"])).round()

# Create one-year ahead forecasts based on Total Revenue Model for the entire
# sample period.
df["tot_rev_nested_pred"] = (
    (
        wline_nested_slope * df["wline_cust_pred"]
        + wless_nested_slope * df["wless_cust_pred"]
    )
    / 1000000
).round(2)

# Create one-year ahead forecasts based on Total Revenue Model for the training
# sample period. This is being done as a robustness check.
df["tot_rev_nested_pred_prior_betas"] = (
    (181.57 * df["wline_cust_pred"] + 215.14 * df["wless_cust_pred"]) / 1000000
).round(2)

# %%
fig, ax = plt.subplots()
ax.scatter(
    df["date"], df["tot_rev_mils"], label="Actual Revenue", color="black"
)
ax.scatter(
    df["date"],
    df["tot_rev_nested_pred"],
    label="Predicted Revenue",
    color="orange",
)
ax.set_title("Actual vs. Predictions")
ax.set_xlabel("Quarter")
ax.set_ylabel("Revenue")
ax.legend()
ax.set_xticklabels(ax.get_xticks(), rotation="vertical")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
plt.savefig("Actual_v_Pred.pdf", bbox_inches="tight")
