#library("LogisticDx")
#library("ucminf")
#library("ordinal")
#library("aod")
library(dplyr)
library('MASS') # polr
library("oglmx") # ologit.reg
library("pscl") # pR2
library('lmtest') # coeftest, lrtest
library("brant") # brant( Brant's test)
library('performance') # r2_mckelvey
library("BaylorEdPsych") # PseudoR2
library("generalhoslem") # lipsitz, logitgof(Hosmer-Lemeshow), pulkrob(Pulkstenis-Robinson)
Sys.setenv(LANG = "en")
options(scipen=100)
setwd('/Users/huyiqing/PycharmProjects/UW_lab/ML_Project/ae')

pixiv <- read.csv('../data/pixiv_tops_lm.csv',header=TRUE, sep=",")
pixiv$rank <- as.integer((pixiv$rank-1) / 10)+1
pixiv$is_comic <- as.factor(pixiv$is_comic)
pixiv$is_Genshin <- as.factor(pixiv$is_Genshin)
pixiv$is_Honkai <- as.factor(pixiv$is_Honkai)
pixiv$views <- scale(pixiv$views)

pixiv$like_rate2 = pixiv$like_rate^2
pixiv$mark_rate2 = pixiv$mark_rate^2
nrow(pixiv)
names(pixiv)

ai <- pixiv[pixiv$is_ai == 1, ]
man <- pixiv[pixiv$is_ai == 0, ]

# because the rank is evenly distribution, so it is ok to use logit and probit
tier<-ai$rank
probit_ai <- polr(ai$rank~like_rate+mark_rate
                  +is_comic+is_Genshin+is_Honkai
                  +comments+top_cnt+date_diff_day
                  +views+mark_rate2+like_rate2, data=ai)

# Intercepts:部分没有用，不用去解释它
# Coefficients:仅需留意这部分，只能解释他们的符号，
# 并且直接对y的第一个和最后一个进行解释
summary(ologit_ai)
# whenever the variable like_rate increases, 
# the probability for the rank tier 1 decreases,and tier 5 increases
# tvalue >2 的话一般pvalue大于5%

coeftest(ologit_ai) 
brant(ologit_ai)

probit_ai<-ologit.reg(as.factor(tier)~like_rate+mark_rate
           +is_comic+is_Genshin+is_Honkai
           +comments+top_cnt+date_diff_day
           +views+mark_rate2+like_rate2, data=ai)
summary(probit_ai)
coeftest(probit_ai)


tier<-man$rank
ologit_man = MASS::polr(as.factor(tier)~like_rate+mark_rate
              +is_comic+is_Genshin+is_Honkai
              +comments+top_cnt+date_diff_day
              +views+mark_rate2+like_rate2,data=man)
# Brant's test is to verify the proportional add assumption
# the function works with polr model results.
# this test concludes which variable is connected to the rejection of a test.
brant(ologit_man)
tier<-man$rank
probit_man<-ologit.reg(as.factor(tier)~like_rate+mark_rate
                      +is_comic+is_Genshin+is_Honkai
                      +comments+top_cnt+date_diff_day
                      +views+mark_rate2+like_rate2, data=man)

summary(probit_man)
coeftest(probit_man)

# the t value is the ratio of Estimate/Error


pR2(ologit_man)
# r2_mckelvey(ologit_ai)
# PseudoR2(ologit_ai)

# Likelihood ratio test
# restricted model with constant only:as.factor(tier)~1
# the likehood ratio test statistic ==-11 5165.5, 
# v-value associated with test statistics is 2.2e-16<0.05
# we have to reject the H0: beta1=beta2=0(which is the condition of restriction), 
# so they jointly significant. so we should choose unrestricted model
ologit_man_restricted <- polr(as.factor(tier)~1, data=man)
lrtest(ologit_man, ologit_man_restricted)

# Goodness-of-fit tests
# the H0 of Lipsitz and logitgof is the form of our model is appropriate
# however Lipsitz is significant, 'logitgof' is not. If one of the is un-appropriate
# we say the model has a problem, we need to correct the model. 
# only if both test get the model is appropriate,
# then we can say the model is appropriate
lipsitz.test(ologit_man) 
logitgof(man$rank, fitted(ologit_man), g=5, ord = TRUE) # Lemeshow

# Pulkstenis-Robinson
# only if we have dummy variable, we can use robinson test.
pulkrob.chisq(ologit_man, c("is_comic"))
pulkrob.deviance(ologit_man, c("is_comic"))



# general-to-specific method to variables selection
# general model

tier<- man$rank
man_all <- ologit_man
summary(man_all)
coeftest(man_all)

omit_genshin <- polr(as.factor(tier)~like_rate+mark_rate
                       +is_comic+is_Honkai
                       +comments+top_cnt+date_diff_day
                       +views+mark_rate2+like_rate2,data=man)
summary(omit_genshin)
coeftest(omit_genshin)
# test whether all insignificant variables all jointly insignificant
anova(man_all, omit_genshin)
# p-value=0.24>0.05 cant reject, so two model is the same, then we can choose a better on with less params
# now all parameters are significant

tier<-ai$rank
ai_all <- ologit_ai
summary(ai_all)
coeftest(ai_all)

omit_honkai <- polr(as.factor(tier)~like_rate+mark_rate
                    +is_comic+is_Genshin
                    +comments+top_cnt+date_diff_day
                    +views+mark_rate2+like_rate2,data=ai)
summary(omit_honkai)
coeftest(omit_honkai)
anova(ai_all, omit_honkai)
# p-value=0.06649932>0.05 cant reject, so two model is the same, then we can choose a better on with less params
# now all parameters are significant

# all insignificant variables are jointly significant
# therefore we have to drop variables in the way one after another
# let's drop "the most insignificant" variable from reg1
# that is Wholesale/Retail trade


