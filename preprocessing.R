setwd("~/fake_news_classifier/data")
dataset = read.csv("fake.csv")
d_subset = dataset[,c(5,6)]
dataset$label= "FAKE"
write.csv(dataset,file="fake.csv")
d2 = read.csv("fake_or_real_news.csv")
dataset = dataset[,c(2,3,4)]
d2 = d2[,c(2,3,4)]
data = data[!(is.na(data$title) | data$title==""), ]