import time
from core.config import settings


class TextBuilder:
    """
    Converts a stream of detected letters into words and sentences.

    Logic:
    - A letter must be stable for LETTER_BUFFER_FRAMES before confirming
    - A pause longer than WORD_PAUSE_SECONDS adds a space (new word)
    - Sentence is built word by word
    - Backspace gesture clears last letter
    - Fist gesture clears entire sentence
    """

    def __init__(self):
        self.current_letter = None
        self.current_word = ""
        self.sentence = ""
        self.last_detection_time = None
        self.last_confirmed_letter = None

        # Stability buffer
        self.letter_buffer = []
        self.buffer_size = settings.LETTER_BUFFER_FRAMES

        # Timing
        self.word_pause = settings.WORD_PAUSE_SECONDS
        self.max_length = settings.MAX_SENTENCE_LENGTH

    def update(self, detected_letter):
        """
        Call this on every frame with the detected letter (or None).
        Returns current state as a dict.
        """
        now = time.time()

        if detected_letter:
            self.current_letter = detected_letter
            self.last_detection_time = now

            # Add to stability buffer
            self.letter_buffer.append(detected_letter)
            if len(self.letter_buffer) > self.buffer_size:
                self.letter_buffer.pop(0)

            # Confirm letter only if buffer is full and all same letter
            if (len(self.letter_buffer) == self.buffer_size and
                    all(l == detected_letter for l in self.letter_buffer)):

                if detected_letter != self.last_confirmed_letter:
                    self._confirm_letter(detected_letter)
                    self.last_confirmed_letter = detected_letter

        else:
            self.current_letter = None
            self.letter_buffer = []
            self.last_confirmed_letter = None

            # Check for word pause
            if (self.last_detection_time and
                    now - self.last_detection_time > self.word_pause and
                    self.current_word):
                self._confirm_word()

        return self.get_state()

    def _confirm_letter(self, letter):
        """Add confirmed letter to current word."""
        if len(self.sentence) + len(self.current_word) < self.max_length:
            self.current_word += letter
            print(f"[TextBuilder] Confirmed letter: {letter} | Word: {self.current_word}")

    def _confirm_word(self):
        """Finalize current word and add to sentence."""
        if self.current_word:
            self.sentence += self.current_word + " "
            print(f"[TextBuilder] Confirmed word: {self.current_word} | Sentence: {self.sentence}")
            self.current_word = ""
            self.last_detection_time = None

    def backspace(self):
        """Remove last letter from current word."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.last_confirmed_letter = None
        elif self.sentence:
            # Remove last word from sentence
            words = self.sentence.rstrip().rsplit(" ", 1)
            self.sentence = words[0] + " " if len(words) > 1 else ""
            self.last_confirmed_letter = None

    def clear(self):
        """Clear everything — triggered by fist gesture."""
        self.current_letter = None
        self.current_word = ""
        self.sentence = ""
        self.letter_buffer = []
        self.last_confirmed_letter = None
        self.last_detection_time = None
        print("[TextBuilder] Cleared.")

    def get_state(self):
        """Return current text builder state."""
        return {
            "current_letter": self.current_letter,
            "current_word": self.current_word,
            "sentence": self.sentence.strip(),
            "buffer_progress": len(self.letter_buffer) / self.buffer_size
        }

    def get_display_sentence(self):
        """Return sentence + current word being built for display."""
        full = self.sentence + self.current_word
        return full.strip() if full.strip() else ""