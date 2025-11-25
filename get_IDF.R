#-------------------------------------------------------------------------------------------------------
# This is the R script used to estimate the IDF curve for selected return periods
# 2, 5, 10, 25, 50, 100, and 500 years
# the GEV distribution is used here
#-------------------------------------------------------------------------------------------------------
#install.packages('lmom', dependencies=TRUE, repos='http://cran.rstudio.com/', lib='./Rlibrary')
library(lmom, lib.loc='./Rlibrary')



# -----------------------------------------------------------------------------
# 90 percent CI for IDF using Gumbel distribution
esti_idf90 <- function(data) {

    # probabilities: 0.50 to 0.99 plus 0.998
    probs <- c(seq(0.50, 0.99, by = 0.01), 0.998)
    nprob <- length(probs)

    # output: each row = probability; col1 = 5%; col2 = 95%
    output <- matrix(NaN, nrow = nprob, ncol = 2)

    # compute L-moments
    lmom_bar <- samlmu(data)
    if (any(is.nan(lmom_bar))) return(output)

    # fit Gumbel parameters
    par1 <- try(pelgum(lmom_bar), silent = TRUE)
    if (inherits(par1, "try-error")) return(output)

    # Monte Carlo matrix for all quantiles
    flow <- matrix(0, nrow = 1000, ncol = nprob)

    # Monte Carlo sampling
    for (i in 1:1000) {

        # resample synthetic data
        resample <- quagum(runif(length(data)), para = c(par1[[1]], par1[[2]]))
        lmom_new <- samlmu(resample)
        par_new <- try(pelgum(lmom_new), silent = TRUE)

        if (inherits(par_new, "try-error")) {
            flow[i, ] <- NaN
        } else {
            # compute all quantiles for this MC sample
            for (j in 1:nprob) {
                flow[i, j] <- quagum(probs[j], para = c(par_new[[1]], par_new[[2]]))
            }
        }
    }

    # final 5%-95% CIs
    for (k in 1:nprob) {
        output[k, 1] <- quantile(flow[, k], 0.05, na.rm = TRUE)
        output[k, 2] <- quantile(flow[, k], 0.95, na.rm = TRUE)
    }

    return(output)
}




# # -----------------------------------------------------------------------------
# # define IDF function
# esti_idf <- function(data) {  # data is a n x 1 vector, detrended annual maximum data

# # 7 x 1, for each return period
# output  <- matrix(data=NaN, nrow=7, ncol=1)

# # fit the GEV distribution and estimate the parameter using L-moments method
# lmom_bar <- samlmu(data)

# # fit the GEV distribution and estimate the parameter using L-moments method
# if (sum(lmom_bar)=="NaN") {
# output <- matrix(data=NaN, nrow=7, ncol=2)
# } else {
# # par <- try(pelgev(lmom_bar))
# par <- try(pelgum(lmom_bar))
# if (class(par) == 'try-error') {
# output <- matrix(data=NaN, nrow=7, ncol=2)

# } else {
# # estimate the quantiles (return periods)
# output[1,1] <- quagev(0.5, para = c(par[[1]], par[[2]], par[[3]]))    # 2-year
# output[2,1] <- quagev(0.8, para = c(par[[1]], par[[2]], par[[3]]))    # 5-year
# output[3,1] <- quagev(0.9, para = c(par[[1]], par[[2]], par[[3]]))    # 10-year
# output[4,1] <- quagev(0.96, para = c(par[[1]], par[[2]], par[[3]]))   # 25-year
# output[5,1] <- quagev(0.98, para = c(par[[1]], par[[2]], par[[3]]))   # 50-year
# output[6,1] <- quagev(0.99, para = c(par[[1]], par[[2]], par[[3]]))   # 100-year
# output[7,1] <- quagev(0.998, para = c(par[[1]], par[[2]], par[[3]]))  # 500-year
# }
# }

# return(output)
# }


# -----------------------------------------------------------------------------
# define IDF function using Gumbel distribution
esti_idf <- function(data) {

    # quantile levels: 0.50 to 0.99 by 0.01, plus 0.998
    probs <- c(seq(0.50, 0.99, by = 0.01), 0.998)

    # prepare output
    output <- matrix(NaN, nrow = length(probs), ncol = 1)

    # compute L-moments
    lmom_bar <- samlmu(data)

    # check if L-moment results are valid
    if (any(is.nan(lmom_bar))) {
        return(output)
    }

    # fit Gumbel distribution using L-moments
    par <- try(pelgum(lmom_bar), silent = TRUE)

    if (inherits(par, "try-error")) {
        return(output)
    }

    # compute quantiles
    for (i in seq_along(probs)) {
        output[i, 1] <- quagum(probs[i], para = c(par[[1]], par[[2]]))
    }

    return(output)
}


# ------------------------------------------------------------------------------
duration = c('24h', '48h', '72h')
variable = c('P', 'W_veg')

for (d in 1:length(duration)) {

for (v in 1:length(variable)) {

    # IDF:
    # Row 1: point estimates
    # Row 2: 5 percent CI
    # Row 3: 95 percent CI
    # All columns: 51 probabilities (0.50 to 0.99 plus 0.998)
    IDF = matrix(data = NaN, nrow = 3, ncol = 51)

    # read annual max file
    file_name1 <- sprintf('./output/am_%s_%s', duration[d], variable[v])
    data <- read.table(file_name1, header = FALSE)

    # remove first year
    data_new <- data[-1,]
    data_am <- data_new[,4]

    # point IDF estimates
    temp <- esti_idf(data_am)     # 51×1
    IDF[1,] <- t(temp)

    # 90 percent CI
    temp2 <- esti_idf90(data_am)  # 51×2
    IDF[2,] <- t(temp2[,1])       # 5 percent
    IDF[3,] <- t(temp2[,2])       # 95 percent

    # save output
    file_name2 <- sprintf('./output/IDF_%s_%s', duration[d], variable[v])
    write.table(IDF, file_name2, row.names = FALSE, col.names = FALSE)
}
}









