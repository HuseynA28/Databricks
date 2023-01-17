# Databricks notebook source
import pandas as pd
from pyspark.sql.functions import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
 
from fbprophet import Prophet

# COMMAND ----------

data = spark.read.table("hive_metastore.default.data12")

# COMMAND ----------


pandasDF = data.toPandas()

# COMMAND ----------

pandasDF.head()

# COMMAND ----------

display(data)

# COMMAND ----------

df=data.select(col("_c0").alias("date"), col("_c1").alias("name"),  col("_c2").alias("price"))

# COMMAND ----------

display(df)

# COMMAND ----------

name_list = ['Bitcoin', 'Ethereum', 'BNB', "Monero", "Quant"]

# COMMAND ----------

df_new = df.toPandas()


# COMMAND ----------

df_new.head()

# COMMAND ----------

df_new.set_index('date', inplace=True)

# COMMAND ----------



# COMMAND ----------

name = 'Bitcoin'
df_name = df[df['name'] == name]['price']

# COMMAND ----------


# Select the specific name value


# Split the data
train, test = train_test_split(df_name, test_size=0.2, shuffle=False)

# Fit the model and make predictions
model = Prophet()
model.fit(train)
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Evaluate the model
rmse = mean_squared_error(test, forecast.yhat[-len(test):])
print(f'RMSE for {name} is {rmse}')

# COMMAND ----------

# MAGIC %fs ls '/databricks-datasest'

# COMMAND ----------

dbutils.help()

# COMMAND ----------

dbutils.fs.help()

# COMMAND ----------

dbutils.fs.ls()

# COMMAND ----------

file=dbutils.fs.ls('/databricks-datasets')

# COMMAND ----------

display(file)

# COMMAND ----------


