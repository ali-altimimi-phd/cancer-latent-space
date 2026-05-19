# ==============================================================================
# File: build_selection_regime_labels.R
# Purpose: Build fully resolved probe-selection regime labels
# Role: Shared helper for structural inference engines
# ==============================================================================

#' Build probe-selection regime labels
#'
#' Converts broad probe-selection families such as "limma" and "variance" into
#' fully resolved regime labels matching filtered-probe file names.
#'
#' Examples:
#'   "limma"    -> "limma"
#'   "variance" + variance_mode = "top_n", variance_top_n = 3000
#'              -> "variance_top3000"
#'   "variance" + variance_mode = "threshold", variance_threshold = 0.75
#'              -> "variance_q075"
#'
#' @param selection_regimes Character vector, e.g. c("limma", "variance").
#' @param variance_mode "top_n" or "threshold".
#' @param variance_top_n Integer top-N value.
#' @param variance_threshold Numeric quantile threshold.
#'
#' @return Character vector of resolved regime labels.
build_selection_regime_labels <- function(selection_regimes,
                                          variance_mode,
                                          variance_top_n,
                                          variance_threshold) {
  regimes <- character(0)
  
  for (selection_method_i in selection_regimes) {
    
    regime_i <- if (identical(selection_method_i, "variance")) {
      
      if (identical(variance_mode, "top_n")) {
        sprintf("variance_top%d", variance_top_n)
      } else if (identical(variance_mode, "threshold")) {
        sprintf(
          "variance_q%s",
          gsub("\\.", "", as.character(variance_threshold))
        )
      } else {
        stop(
          sprintf("Unknown variance_mode: %s", variance_mode),
          call. = FALSE
        )
      }
      
    } else if (identical(selection_method_i, "limma")) {
      
      "limma"
      
    } else {
      stop(
        sprintf("Unknown selection_regime: %s", selection_method_i),
        call. = FALSE
      )
    }
    
    regimes <- c(regimes, regime_i)
  }
  
  unique(regimes)
}