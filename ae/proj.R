library(zoo)
library(xts)

setwd('/Users/huyiqing/PycharmProjects/UW_lab/ML_Project/ae')
Sys.setenv(LANG = "en")
options(scipen=100)

# lm
pixiv <- read.csv('../data/pixiv_tops_lm.csv',header=TRUE, sep=",")
pixiv <- pixiv[, !(names(pixiv) %in% c('pid', 'date', 'like_rate'))]
head(pixiv)
pixiv_all = lm(mark_rate~comments+views+rank+top_cnt+date_diff_day+
            as.factor(is_comic)+as.factor(is_ai)+as.factor(is_Genshin)+as.factor(is_Honkai), 
          data=pixiv)
summary(pixiv_all)


# time series
pixiv <- read.csv('../data/pixiv_views_ts.csv',header=TRUE, sep=",")
head(pixiv)
tail(pixiv)

pixiv$date <- as.Date(pixiv$date, format = "%Y-%m-%d")
pixiv <- xts(pixiv[,-1], # data columns (without a column with date)
                 pixiv$date) # date/time index
names(pixiv)
plot(pixiv$mean_views_origin,
     main = "Average views of manmade works in Top 50")

plot(pixiv$mean_views_ai,
     main = "Average views of AI works in Top 50")


pixiv.zoo <- as.zoo(pixiv) 
plot(pixiv.zoo)

colors_ <- c("blue", "green")
plot(pixiv.zoo, 
     plot.type = "single",
     col = colors_,
     ylab = "views", 
     xlab = "date",
     main ='Average views of works in Top 50')


legend("topright",     # legend position - combination of top, bottom and left, right
       names(pixiv.zoo), # legend elements (names of the series)
       text.col = colors_) 

save(list = "pixiv", file = "pixiv.RData")



