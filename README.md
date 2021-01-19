# CommonCrawl-Coding-Task
This project is for the data science internship

In this project, I use a naive idea, i.e., key words related to covid-19 and economic impact should at least have key words like 'covid / covid-19 / sars', and 'economy / finance', etc. I separate those keywords to 3 types: covid-related words, economy related words, and tendency words(like 'increase', 'decrease', etc). I require key words of each type should at least show up more than once to increase accuracy. 

Due to the time and computational resource limit, I cannot make a more delicate improvement. If time allow, I will try to collect some reports on covid and economy impact, train a logistic regression model on the labelled reports(preprocessing by word-to-vector), and filter the new documents by this logistic regression model. 

Due to computational resource limit, I only use data on October, 2020 (2020-45)
