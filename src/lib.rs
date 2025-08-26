use anyhow::Result;
use kalosm::language::*;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tracing::{info, warn, error, debug, trace};
use std::time::Instant;

/// Maximum number of retry attempts for script generation and execution
const MAX_RETRIES: usize = 3;

/// A client that holds the AI model for dynamically generating parsing scripts.
pub struct ParserClient {
    model: Llama,
}

#[derive(Debug)]
pub struct ParseAttempt {
    attempt_number: usize,
    script: String,
    error: Option<String>,
    success: bool,
}

impl ParserClient {
    /// Creates a new `ParserClient` and loads the AI model.
    pub async fn new() -> Result<Self> {
        let start_time = Instant::now();
        info!("Starting ParserClient initialization...");
        info!("Using TinyLlama 1.1B Chat model for faster performance");
        
        debug!("Building Llama model with TinyLlama source...");
        let model = Llama::builder()
            .with_source(LlamaSource::tiny_llama_1_1b_chat()) // Use the chat version which has correct URL format
            .build()
            .await?;
        
        let elapsed = start_time.elapsed();
        info!("✅ ParserClient initialized successfully in {:.2}s", elapsed.as_secs_f64());
        
        Ok(Self { model })
    }

    /// Dynamically parses a document using an AI-generated Python script with retry logic.
    pub async fn dynamic_parse(&self, document: &str, instructions: &str) -> Result<String> {
        let overall_start = Instant::now();
        info!("🔄 Starting dynamic parse operation");
        info!("📄 Document length: {} characters", document.len());
        info!("📝 Instructions: {}", instructions);
        
        debug!("Creating chat session with system prompt...");
        let mut chat = self.model.chat().with_system_prompt(self.get_system_prompt());
        let mut attempts: Vec<ParseAttempt> = Vec::new();
        
        for attempt in 1..=MAX_RETRIES {
            let attempt_start = Instant::now();
            info!("🎯 Parsing attempt {}/{}", attempt, MAX_RETRIES);
            
            debug!("Building user prompt for attempt {}...", attempt);
            let user_prompt = self.build_user_prompt(document, instructions, &attempts, attempt);
            trace!("User prompt length: {} characters", user_prompt.len());
            
            // Generate the script
            info!("🤖 Generating Python script with AI model...");
            let script_gen_start = Instant::now();
            let python_script = match chat.add_message(&user_prompt).await {
                Ok(script) => {
                    let gen_elapsed = script_gen_start.elapsed();
                    info!("✅ Script generated successfully in {:.2}s", gen_elapsed.as_secs_f64());
                    debug!("Generated script length: {} characters", script.len());
                    trace!("Generated script preview: {}", 
                        script.chars().take(200).collect::<String>().replace('\n', "\\n"));
                    script
                },
                Err(e) => {
                    let gen_elapsed = script_gen_start.elapsed();
                    let error_msg = format!("Failed to generate script: {}", e);
                    error!("❌ Script generation failed after {:.2}s: {}", gen_elapsed.as_secs_f64(), error_msg);
                    
                    attempts.push(ParseAttempt {
                        attempt_number: attempt,
                        script: String::new(),
                        error: Some(error_msg.clone()),
                        success: false,
                    });
                    
                    if attempt == MAX_RETRIES {
                        let total_elapsed = overall_start.elapsed();
                        error!("💥 All script generation attempts failed after {:.2}s", total_elapsed.as_secs_f64());
                        anyhow::bail!("Failed to generate script after {} attempts. Last error: {}", MAX_RETRIES, error_msg);
                    }
                    continue;
                }
            };

            // Execute the script
            info!("🐍 Executing Python script...");
            let exec_start = Instant::now();
            match self.execute_python_script(&python_script, document).await {
                Ok(result) => {
                    let exec_elapsed = exec_start.elapsed();
                    let attempt_elapsed = attempt_start.elapsed();
                    let total_elapsed = overall_start.elapsed();
                    
                    info!("🎉 Successfully parsed document on attempt {}", attempt);
                    info!("⏱️  Execution time: {:.2}s, Attempt time: {:.2}s, Total time: {:.2}s", 
                        exec_elapsed.as_secs_f64(), attempt_elapsed.as_secs_f64(), total_elapsed.as_secs_f64());
                    info!("📊 Result length: {} characters", result.len());
                    debug!("Result preview: {}", result.chars().take(200).collect::<String>());
                    
                    attempts.push(ParseAttempt {
                        attempt_number: attempt,
                        script: python_script,
                        error: None,
                        success: true,
                    });
                    return Ok(result);
                }
                Err(e) => {
                    let exec_elapsed = exec_start.elapsed();
                    let attempt_elapsed = attempt_start.elapsed();
                    let error_msg = format!("Script execution failed: {}", e);
                    
                    warn!("⚠️  Attempt {} failed after {:.2}s (exec: {:.2}s): {}", 
                        attempt, attempt_elapsed.as_secs_f64(), exec_elapsed.as_secs_f64(), error_msg);
                    debug!("Failed script content: {}", python_script);
                    
                    attempts.push(ParseAttempt {
                        attempt_number: attempt,
                        script: python_script,
                        error: Some(error_msg.clone()),
                        success: false,
                    });
                    
                    if attempt == MAX_RETRIES {
                        let total_elapsed = overall_start.elapsed();
                        error!("💥 All parsing attempts failed after {:.2}s", total_elapsed.as_secs_f64());
                        anyhow::bail!(
                            "All {} parsing attempts failed. Final error: {}\n\nAll attempts:\n{}", 
                            MAX_RETRIES, 
                            error_msg,
                            self.format_attempt_history(&attempts)
                        );
                    }
                }
            }
        }
        
        unreachable!("Should have returned or failed within the retry loop")
    }

    /// Executes a Python script with the given document as input
    async fn execute_python_script(&self, python_script: &str, document: &str) -> Result<String> {
        let start_time = Instant::now();
        debug!("🐍 Starting Python script execution...");
        debug!("Script size: {} bytes, Document size: {} bytes", python_script.len(), document.len());
        
        trace!("Spawning python3 process...");
        let mut cmd = Command::new("python3")
            .arg("-c")
            .arg(python_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        debug!("Writing document to stdin...");
        let mut stdin = cmd.stdin.take().expect("Failed to open stdin");
        let document_for_script = document.to_string();
        
        tokio::spawn(async move {
            if let Err(e) = stdin.write_all(document_for_script.as_bytes()).await {
                error!("Failed to write to stdin: {}", e);
            } else {
                trace!("Successfully wrote document to stdin");
            }
        });

        debug!("Waiting for Python process to complete...");
        let output = cmd.wait_with_output().await?;
        let exec_elapsed = start_time.elapsed();
        
        debug!("Python process completed in {:.3}s", exec_elapsed.as_secs_f64());
        debug!("Exit status: {:?}", output.status);
        debug!("Stdout length: {} bytes", output.stdout.len());
        debug!("Stderr length: {} bytes", output.stderr.len());

        if output.status.success() {
            trace!("Python script executed successfully");
            let stdout = String::from_utf8(output.stdout)?;
            
            // Validate that we got some meaningful output
            if stdout.trim().is_empty() {
                warn!("Script executed successfully but produced no output");
                anyhow::bail!("Script executed successfully but produced no output");
            }
            
            debug!("Validating JSON output...");
            // Try to validate it's valid JSON
            if let Err(e) = serde_json::from_str::<serde_json::Value>(&stdout) {
                error!("Script output is not valid JSON: {}", e);
                debug!("Invalid JSON output: {}", stdout);
                anyhow::bail!("Script output is not valid JSON: {}\nOutput was: {}", e, stdout);
            }
            
            info!("✅ Script executed successfully and produced valid JSON");
            Ok(stdout)
        } else {
            let error_message = String::from_utf8(output.stderr)?;
            error!("Python script execution failed with exit code: {}", output.status.code().unwrap_or(-1));
            error!("STDERR: {}", error_message);
            debug!("Failed script:\n{}", python_script);
            
            anyhow::bail!(
                "Python script execution failed with exit code: {}\nSTDERR: {}\nSCRIPT:\n{}", 
                output.status.code().unwrap_or(-1),
                error_message,
                python_script
            );
        }
    }

    /// Gets the system prompt for the AI model
    fn get_system_prompt(&self) -> &'static str {
        debug!("Using system prompt for AI model");
        r#"
You are an expert Python programmer that creates parsing scripts. Your task is to write a single, complete Python script based on the user's request.

CRITICAL RULES:
1. The script you write will receive the raw document text via standard input (stdin).
2. The script must print a single, valid JSON object to standard output (stdout).
3. The script MUST NOT use any external libraries like BeautifulSoup. Use only standard libraries like `sys`, `json`, and `re`.
4. Your output must be ONLY the raw Python code. Do not include explanations, markdown, or code blocks.
5. Always include proper error handling to avoid crashes.
6. If you cannot find the requested data, return an empty JSON object {} rather than failing.
7. Make sure your JSON output is properly formatted and valid.

If this is a retry attempt, learn from the previous errors and fix them in your new script.
"#
    }

    /// Builds the user prompt, including error history for retry attempts
    fn build_user_prompt(&self, document: &str, instructions: &str, attempts: &[ParseAttempt], current_attempt: usize) -> String {
        debug!("Building user prompt for attempt {}", current_attempt);
        
        let mut prompt = format!(
            r#"
**Instructions:**
{}

**Document to Parse:**
---
{}
---
"#,
            instructions, document
        );

        // Add error history for retry attempts
        if current_attempt > 1 && !attempts.is_empty() {
            debug!("Adding error history from {} previous attempts", attempts.len());
            prompt.push_str("\n**Previous Attempts and Errors:**\n");
            for attempt in attempts {
                prompt.push_str(&format!("Attempt {}: ", attempt.attempt_number));
                if let Some(error) = &attempt.error {
                    debug!("Including error from attempt {}: {}", attempt.attempt_number, error);
                    prompt.push_str(&format!("FAILED - {}\n", error));
                    if !attempt.script.is_empty() {
                        prompt.push_str("Script that failed:\n```python\n");
                        prompt.push_str(&attempt.script);
                        prompt.push_str("\n```\n\n");
                    }
                } else {
                    prompt.push_str("SUCCESS\n");
                }
            }
            prompt.push_str("Please learn from these errors and create a better script.\n\n");
        }

        prompt.push_str("Provide the Python script now:");
        trace!("Final prompt length: {} characters", prompt.len());
        prompt
    }

    /// Formats the attempt history for error reporting
    fn format_attempt_history(&self, attempts: &[ParseAttempt]) -> String {
        debug!("Formatting attempt history for {} attempts", attempts.len());
        let mut history = String::new();
        for attempt in attempts {
            history.push_str(&format!("--- Attempt {} ---\n", attempt.attempt_number));
            if attempt.success {
                history.push_str("Status: SUCCESS\n");
            } else {
                history.push_str("Status: FAILED\n");
                if let Some(error) = &attempt.error {
                    history.push_str(&format!("Error: {}\n", error));
                }
            }
            if !attempt.script.is_empty() {
                history.push_str("Script:\n");
                history.push_str(&attempt.script);
                history.push_str("\n\n");
            }
        }
        history
    }

    /// Alternative method that returns detailed attempt information along with the result
    pub async fn dynamic_parse_with_details(&self, document: &str, instructions: &str) -> Result<(String, Vec<ParseAttempt>)> {
        let overall_start = Instant::now();
        info!("🔄 Starting dynamic parse with details");
        info!("📄 Document length: {} characters", document.len());
        info!("📝 Instructions: {}", instructions);
        
        debug!("Creating chat session with system prompt...");
        let mut chat = self.model.chat().with_system_prompt(self.get_system_prompt());
        let mut attempts: Vec<ParseAttempt> = Vec::new();
        
        for attempt in 1..=MAX_RETRIES {
            let attempt_start = Instant::now();
            info!("🎯 Parsing attempt {}/{}", attempt, MAX_RETRIES);
            
            debug!("Building user prompt for attempt {}...", attempt);
            let user_prompt = self.build_user_prompt(document, instructions, &attempts, attempt);
            
            // Generate the script
            info!("🤖 Generating Python script with AI model...");
            let script_gen_start = Instant::now();
            let python_script = match chat.add_message(&user_prompt).await {
                Ok(script) => {
                    let gen_elapsed = script_gen_start.elapsed();
                    info!("✅ Script generated successfully in {:.2}s", gen_elapsed.as_secs_f64());
                    debug!("Generated script length: {} characters", script.len());
                    script
                },
                Err(e) => {
                    let gen_elapsed = script_gen_start.elapsed();
                    let error_msg = format!("Failed to generate script: {}", e);
                    error!("❌ Script generation failed after {:.2}s: {}", gen_elapsed.as_secs_f64(), error_msg);
                    
                    attempts.push(ParseAttempt {
                        attempt_number: attempt,
                        script: String::new(),
                        error: Some(error_msg.clone()),
                        success: false,
                    });
                    
                    if attempt == MAX_RETRIES {
                        let total_elapsed = overall_start.elapsed();
                        error!("💥 All script generation attempts failed after {:.2}s", total_elapsed.as_secs_f64());
                        anyhow::bail!("Failed to generate script after {} attempts. Last error: {}", MAX_RETRIES, error_msg);
                    }
                    continue;
                }
            };

            // Execute the script
            info!("🐍 Executing Python script...");
            let exec_start = Instant::now();
            match self.execute_python_script(&python_script, document).await {
                Ok(result) => {
                    let exec_elapsed = exec_start.elapsed();
                    let attempt_elapsed = attempt_start.elapsed();
                    let total_elapsed = overall_start.elapsed();
                    
                    info!("🎉 Successfully parsed document on attempt {}", attempt);
                    info!("⏱️  Execution time: {:.2}s, Attempt time: {:.2}s, Total time: {:.2}s", 
                        exec_elapsed.as_secs_f64(), attempt_elapsed.as_secs_f64(), total_elapsed.as_secs_f64());
                    info!("📊 Result length: {} characters", result.len());
                    debug!("Result preview: {}", result.chars().take(200).collect::<String>());
                    
                    attempts.push(ParseAttempt {
                        attempt_number: attempt,
                        script: python_script,
                        error: None,
                        success: true,
                    });
                    return Ok((result, attempts));
                }
                Err(e) => {
                    let exec_elapsed = exec_start.elapsed();
                    let attempt_elapsed = attempt_start.elapsed();
                    let error_msg = format!("Script execution failed: {}", e);
                    
                    warn!("⚠️  Attempt {} failed after {:.2}s (exec: {:.2}s): {}", 
                        attempt, attempt_elapsed.as_secs_f64(), exec_elapsed.as_secs_f64(), error_msg);
                    debug!("Failed script content: {}", python_script);
                    
                    attempts.push(ParseAttempt {
                        attempt_number: attempt,
                        script: python_script,
                        error: Some(error_msg.clone()),
                        success: false,
                    });
                    
                    if attempt == MAX_RETRIES {
                        let total_elapsed = overall_start.elapsed();
                        error!("💥 All parsing attempts failed after {:.2}s", total_elapsed.as_secs_f64());
                        anyhow::bail!(
                            "All {} parsing attempts failed. Final error: {}\n\nAll attempts:\n{}", 
                            MAX_RETRIES, 
                            error_msg,
                            self.format_attempt_history(&attempts)
                        );
                    }
                }
            }
        }
        
        unreachable!("Should have returned or failed within the retry loop")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::OnceLock;

    // This static variable will ensure the initialization logic is run only once.
    static TRACING: OnceLock<()> = OnceLock::new();

    /// A helper function to safely initialize the tracing subscriber.
    fn setup_tracing() {
        TRACING.get_or_init(|| {
            // This closure will only be executed the first time 
            // `setup_tracing` is called.
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::DEBUG) // Set to DEBUG for more detailed logging
                .with_target(false)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .init();
        });
    }

    #[tokio::test]
    async fn test_successful_parse() {
        // Call the setup function at the beginning of each test.
        setup_tracing();
        
        println!("🧪 Starting test_successful_parse");
        info!("Test: test_successful_parse started");
        
        println!("Initializing parser client...");
        let client = ParserClient::new().await.expect("Failed to get parser");
        println!("Client initialized.");

        let html_document = r#"
    <body>
        <div class="product"><h1>Super Toaster 5000</h1><p>Price: $49.99</p></div>
    </body>
    "#;
        let parsing_instructions = "Extract the product name and its price as a float.";

        println!("Parsing document...");
        info!("Starting document parsing test");
        
        match client
            .dynamic_parse(html_document, parsing_instructions)
            .await
        {
            Ok(json_output) => {
                println!("\n✅ Success! Parsed JSON:");
                println!("{}", json_output);
                info!("Test completed successfully");
            }
            Err(e) => {
                error!("Test failed: {}", e);
                panic!("\n❌ An error occurred: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_parse_with_details() {
        setup_tracing();
        
        println!("🧪 Starting test_parse_with_details");
        info!("Test: test_parse_with_details started");
        
        let client = ParserClient::new().await.expect("Failed to get parser");

        let html_document = r#"<div><span>Item: Widget</span><span>Cost: $25.50</span></div>"#;
        let parsing_instructions = "Extract the item name and cost as separate fields.";

        info!("Starting detailed parsing test");
        match client
            .dynamic_parse_with_details(html_document, parsing_instructions)
            .await
        {
            Ok((result, attempts)) => {
                println!("✅ Success! Result: {}", result);
                println!("Total attempts: {}", attempts.len());
                info!("Test completed with {} attempts", attempts.len());
                
                for attempt in attempts {
                    println!("Attempt {}: {}", attempt.attempt_number, if attempt.success { "SUCCESS" } else { "FAILED" });
                    if let Some(error) = &attempt.error {
                        debug!("Attempt {} error: {}", attempt.attempt_number, error);
                    }
                }
            }
            Err(e) => {
                error!("Detailed parsing test failed: {}", e);
                println!("❌ Failed: {}", e);
            }
        }
    }

    #[tokio::test] 
    async fn test_retry_logic_with_malformed_document() {
        setup_tracing();

        println!("🧪 Starting test_retry_logic_with_malformed_document");
        info!("Test: test_retry_logic_with_malformed_document started");
        
        let client = ParserClient::new().await.expect("Failed to get parser");

        // Intentionally malformed/difficult document
        let html_document = r#"<html><body><div>No clear structure here</div></body></html>"#;
        let parsing_instructions = "Extract the product name and price (this will likely fail initially).";

        info!("Starting retry logic test with malformed document");
        match client
            .dynamic_parse_with_details(html_document, parsing_instructions)
            .await
        {
            Ok((result, attempts)) => {
                println!("✅ Eventually succeeded: {}", result);
                println!("Required {} attempts", attempts.len());
                info!("Retry test succeeded after {} attempts", attempts.len());
            }
            Err(e) => {
                warn!("Retry test failed as expected: {}", e);
                println!("❌ Failed after all retries: {}", e);
            }
        }
    }
}
