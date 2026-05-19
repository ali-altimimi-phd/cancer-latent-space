#' Build ExpressionSet Objects from CEL Files
#'
#' Reads and normalizes CEL files by chip type using RMA, and builds a list
#' of chip-specific `ExpressionSet` objects. Skips unreadable or unidentifiable
#' CEL files and logs them.
#'
#' @param cel_dir Directory containing raw CEL files.
#' @param log_dir Directory to write skipped CEL log files.
#' @param logger Logging object (optional).
#'
#' @return A named list of `ExpressionSet` objects (one per chip type).
#' @export
build_expression_sets <- function(cel_dir, log_dir, logger = NULL) {
  # ---- Dependency Check ----
  if (!requireNamespace("affy", quietly = TRUE)) {
    stop("❌ Package 'affy' is required but not installed.")
  }
  
  if (!requireNamespace("Biobase", quietly = TRUE)) {
    stop("❌ Package 'Biobase' is required but not installed.")
  }
  
  # ---- Ensure Log Directory Exists ----
  dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
  
  # ---- Discover CEL Files ----
  cel_files <- list.files(cel_dir, pattern = "\\.CEL$", full.names = TRUE)
  
  if (length(cel_files) == 0) {
    stop(glue::glue("❌ No CEL files found in: {cel_dir}"))
  }
  
  if (!is.null(logger)) {
    logger$log(glue::glue("📦 Found {length(cel_files)} CEL files."))
  }
  
  # ---- Detect Chip Type ----
  get_chip_type <- function(file) {
    tryCatch({
      ab <- suppressWarnings(affy::read.affybatch(filenames = file))
      Biobase::annotation(ab)
    }, error = function(e) {
      msg <- glue::glue("❌ Failed chip detection: {basename(file)} — {e$message}")
      message(msg)
      if (!is.null(logger)) logger$log(msg)
      NA_character_
    })
  }
  
  chip_types <- vapply(cel_files, get_chip_type, character(1))
  valid_idx  <- !is.na(chip_types)
  
  valid_cels  <- cel_files[valid_idx]
  chip_types  <- chip_types[valid_idx]
  skipped_cels <- cel_files[!valid_idx]
  
  if (length(valid_cels) == 0) {
    stop("❌ No valid CEL files found after chip detection.")
  }
  
  # ---- Log CELs Skipped During Chip Detection ----
  skipped_chiptype_log <- file.path(log_dir, "skipped_chiptype.txt")
  writeLines(basename(skipped_cels), con = skipped_chiptype_log)
  
  if (length(skipped_cels) > 0) {
    msg <- glue::glue("📄 Skipped {length(skipped_cels)} CELs during chip detection — see: {skipped_chiptype_log}")
    message(msg)
    if (!is.null(logger)) logger$log(msg)
  } else {
    if (!is.null(logger)) logger$log("✅ No CEL files were skipped during chip detection.")
  }
  
  # ---- Group Valid CELs by Chip Type ----
  cel_by_chip <- split(valid_cels, chip_types)
  eset_list <- list()
  
  # ---- Build ExpressionSet per Chip ----
  for (chip in names(cel_by_chip)) {
    chip_cels <- cel_by_chip[[chip]]
    
    if (!is.null(logger)) {
      logger$log(glue::glue("🔬 {chip}: {length(chip_cels)} CEL files detected."))
    }
    
    # Validate that each CEL can actually be read
    valid_flags <- logical(length(chip_cels))
    
    for (i in seq_along(chip_cels)) {
      file <- chip_cels[i]
      
      valid_flags[i] <- tryCatch({
        invisible(affy::read.affybatch(filenames = file))
        TRUE
      }, error = function(e) {
        msg <- glue::glue("❌ Skipping {basename(file)} — {e$message}")
        message(msg)
        if (!is.null(logger)) logger$log(msg)
        FALSE
      })
    }
    
    valid_chip_cels <- chip_cels[valid_flags]
    invalid_chip_cels <- chip_cels[!valid_flags]
    
    chip_log <- file.path(log_dir, glue::glue("skipped_{chip}_cel.txt"))
    writeLines(basename(invalid_chip_cels), con = chip_log)
    
    if (length(invalid_chip_cels) > 0) {
      if (!is.null(logger)) {
        logger$log(glue::glue("📄 {chip}: Skipped {length(invalid_chip_cels)} unreadable CEL files — see: {chip_log}"))
      }
    } else {
      if (!is.null(logger)) {
        logger$log(glue::glue("✅ {chip}: No unreadable CEL files detected."))
      }
    }
    
    if (length(valid_chip_cels) == 0) {
      msg <- glue::glue("⚠️ No valid CELs remaining for {chip}. Skipping normalization.")
      message(msg)
      if (!is.null(logger)) logger$log(msg)
      next
    }
    
    if (!is.null(logger)) {
      logger$log(glue::glue("🧪 {chip}: Normalizing {length(valid_chip_cels)} CELs with RMA..."))
    }
    
    raw_affy <- affy::ReadAffy(filenames = valid_chip_cels)
    eset <- affy::rma(raw_affy)
    
    eset_list[[chip]] <- eset
    
    if (!is.null(logger)) {
      logger$log(glue::glue("✅ {chip}: ExpressionSet created successfully."))
    }
  }
  
  # ---- Final Check ----
  if (length(eset_list) == 0) {
    stop("❌ No ExpressionSet objects were created.")
  }
  
  if (!is.null(logger)) {
    logger$log(glue::glue("🏁 ExpressionSet build complete: {length(eset_list)} chip type(s) processed successfully."))
  }
  
  return(eset_list)
}