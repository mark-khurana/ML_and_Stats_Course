# setup.R - Common functions and theme for the course
# Source this file at the start of each chapter

library(tidyverse)

# Course ggplot theme
theme_course <- function(base_size = 13) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 2),
      plot.subtitle = element_text(colour = "grey40"),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom"
    )
}

theme_set(theme_course())

# Load course datasets
load_course_data <- function(name) {
  path <- here::here("data", paste0(name, ".csv"))
  if (!file.exists(path)) {
    path <- file.path("data", paste0(name, ".csv"))
  }
  readr::read_csv(path, show_col_types = FALSE)
}

# Colour palette (colour-blind friendly)
course_colours <- c(
  "#2c6fbb",  # blue

"#e69f00",  # orange
  "#009e73",  # green
  "#cc79a7",  # pink
  "#56b4e9",  # light blue
  "#d55e00",  # red-orange
  "#0072b2",  # dark blue
  "#f0e442"   # yellow
)

scale_colour_course <- function(...) {
  scale_colour_manual(values = course_colours, ...)
}

scale_fill_course <- function(...) {
  scale_fill_manual(values = course_colours, ...)
}
