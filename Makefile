# MOSS-TTS-Nano C inference (safetensors + SIMD). Prompt text: C++ + libsentencepiece (no Python).
CC ?= gcc
CXX ?= g++
ifeq ($(OS),Windows_NT)
  # Some Windows environments export CC=cc/CXX=c++ even when only MinGW gcc/g++ exist.
  ifeq ($(CC),cc)
    CC := gcc
  endif
  ifeq ($(CXX),c++)
    CXX := g++
  endif
endif
CFLAGS ?= -Wall -Wextra -O3 -march=native -ffast-math
CXXFLAGS ?= $(CFLAGS)

BLAS_OK := $(shell ldconfig -p 2>/dev/null | grep -Eq "lib(openblas|blas)" && echo 1 || echo 0)

ifeq ($(BLAS_OK),1)
  CFLAGS += -DMOSS_USE_CBLAS
  CXXFLAGS += -DMOSS_USE_CBLAS
  # cblas.h + consistent linkage with BLAS_LIBS below
  CFLAGS += $(shell pkg-config --cflags openblas 2>/dev/null)
endif

SP_CXXFLAGS := $(shell pkg-config --cflags sentencepiece 2>/dev/null)
SP_LIBS := $(shell pkg-config --libs sentencepiece 2>/dev/null | sed 's/-lsentencepiece_train//g')
ifeq ($(strip $(SP_LIBS)),)
  SP_LIBS = -lsentencepiece
endif

# Only link BLAS when MOSS_USE_CBLAS is enabled (see BLAS_OK above).
BLAS_LIBS :=
ifeq ($(BLAS_OK),1)
  BLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)
  ifeq ($(strip $(BLAS_LIBS)),)
    ifneq ($(shell ldconfig -p 2>/dev/null | grep -E -c 'libopenblas\.so'),0)
      BLAS_LIBS := -lopenblas
    else
      BLAS_LIBS := -lblas
    endif
  endif
endif

LDLIBS = -lm

SRCS = main.c moss_tts.c safetensors.c moss_config.c moss_kernel.c moss_gpt2.c moss_weights.c moss_audio_tok_load.c moss_audio_tok_graph.c moss_audio_tok_encode.c moss_audio_tok_decode.c
OBJS = $(SRCS:.c=.o) moss_sp_prompt.o
TARGET = moss_tts

.PHONY: all clean run-example

ifeq ($(OS),Windows_NT)
  CLEAN_CMD = powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue *.o,$(TARGET),$(TARGET).exe; exit 0"
else
  CLEAN_CMD = rm -f $(OBJS) $(TARGET)
endif

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(SP_LIBS) $(BLAS_LIBS) $(LDLIBS)

moss_sp_prompt.o: moss_sp_prompt.cc moss_sp_prompt.h moss_config.h
	$(CXX) $(CXXFLAGS) -std=c++17 $(SP_CXXFLAGS) -c -o $@ $<

%.o: %.c moss_tts.h moss_audio_tok.h moss_config.h moss_weights.h moss_gpt2.h moss_kernel.h moss_sp_prompt.h safetensors.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	$(CLEAN_CMD)

# Default weight dir: sibling ../weight under MOSS-TTS-Nano
run-example: $(TARGET)
	./$(TARGET) --model-dir ../weight --text "Hello MOSS." --out /tmp/moss_cpp.wav --frames 32
