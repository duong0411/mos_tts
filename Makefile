# MOSS-TTS-Nano C inference (safetensors + SIMD). Prompt text: C++ + libsentencepiece (no Python).
CC ?= gcc
CXX ?= g++
CFLAGS ?= -Wall -Wextra -O3 -march=native -ffast-math
CXXFLAGS ?= $(CFLAGS)

BLAS_OK := $(shell ldconfig -p 2>/dev/null | grep -Eq "lib(openblas|blas)" && echo 1 || echo 0)

ifeq ($(BLAS_OK),1)
  CFLAGS += -DMOSS_USE_CBLAS
  CXXFLAGS += -DMOSS_USE_CBLAS
endif

SP_CXXFLAGS := $(shell pkg-config --cflags sentencepiece 2>/dev/null)
SP_LIBS := $(shell pkg-config --libs sentencepiece 2>/dev/null | sed 's/-lsentencepiece_train//g')
ifeq ($(strip $(SP_LIBS)),)
  SP_LIBS = -lsentencepiece
endif

BLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(strip $(BLAS_LIBS)),)
  BLAS_LIBS = -lblas
endif

LDLIBS = -lm

SRCS = main.c moss_tts.c safetensors.c moss_config.c moss_kernel.c moss_gpt2.c moss_weights.c moss_audio_tok.c
OBJS = $(SRCS:.c=.o) moss_sp_prompt.o
TARGET = moss_tts

.PHONY: all clean run-example

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(SP_LIBS) $(BLAS_LIBS) $(LDLIBS)

moss_sp_prompt.o: moss_sp_prompt.cc moss_sp_prompt.h moss_config.h
	$(CXX) $(CXXFLAGS) -std=c++17 $(SP_CXXFLAGS) -c -o $@ $<

%.o: %.c moss_tts.h moss_audio_tok.h moss_config.h moss_weights.h moss_gpt2.h moss_kernel.h moss_sp_prompt.h safetensors.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

# Default weight dir: sibling ../weight under MOSS-TTS-Nano
run-example: $(TARGET)
	./$(TARGET) --model-dir ../weight --text "Hello MOSS." --out /tmp/moss_cpp.wav --frames 32
