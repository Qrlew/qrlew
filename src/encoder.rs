//! # Encoding utilities
//! 
//! Some simple utilities to encode an integer to string
//! 

pub const BASE_62: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const BASE_63: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";
pub const BASE_64: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-";
pub const BASE_36: &str = "0123456789abcdefghijklmnopqrstuvwxyz";
pub const BASE_37: &str = "0123456789abcdefghijklmnopqrstuvwxyz_";

pub struct Encoder {
    alphabet: Vec<char>,
    length: usize,
}

impl Encoder {
    pub fn new(alphabet: &str, length: usize) -> Encoder {
        let alphabet: Vec<char> = alphabet.chars().collect();
        Encoder { alphabet, length }
    }

    pub fn encode(&self, mut input: u64) -> String {
        let mut result: Vec<char> = Vec::new();
        for _ in 0..self.length {
            result.push(self.alphabet[input as usize % self.alphabet.len()]);
            input /= self.alphabet.len() as u64;
        }
        result.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_encoder() {
        let encoder = Encoder::new(BASE_36, 5);
        let n = 6785678567887567856u64;
        println!("{} -> {}", n, encoder.encode(n));
    }
}
