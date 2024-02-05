function show_qual_answer(button_id, text_id, text){

    const button = document.getElementById(button_id);
    const output = document.getElementById(text_id);

    button.addEventListener("click", function() {
        output.innerHTML = text;
        output.style.display = "";
        output.style.borderStyle = "dotted";
        output.style.borderColor = "orange";
        output.style.padding = "10px";
    });
}

function show_instruction() {
	const messageContainer = document.getElementById("instruction_page");
	messageContainer.innerHTML = `

    <p>
	Thank you for participating in this study. <b style='color:red'> Please read the instruction below thoroughly and carefully.</b>
	</p>

    <br>

    <h2>Instruction</h2>
    <hr>
		<p>
		In this study, you will be shown <b>a pair of answers generated by two AI systems A and B</b>, in addition to an instruction question and a reference sentence.
		</p>
		<p> For each question, your task is to <b>choose one answer </b> between the two systems in terms of the following point:
		<ul>
			<li>which system's answer <mark><b>aligns better</b></mark> and <mark><b> coherent </b></mark> with the instruction and reference sentences. (Please note that <b style="color:green">the reference answer is one possible answer to the instruction question.</b>) </li>
		</ul>
    <br>
    <h2> Example Set</h2>
    <hr>
        <p> Here is one example instruction with a pair of answers by systems A and B. 

        <div id="example_template" style="border-style: dotted; border-color: CornflowerBlue; padding: 10px; color:black">
            <p>### <b> Question: </b> Which system's answer is <b style='color:SlateBlue'>more coherent</b>, considering the reference and instruction sentences? </p>
            <ul>
                <li> <b>The instruction</b>: What are the simple ways to Adopt a Healthy and sustainable Eating Pattern? </li>
                <li> <b>The reference</b>: Include more whole foods in your diet.</li>
            </ul>

            <div id="systemA" class="mt-3" style="text-align:center;">
                <p> First, Double-Click <b>System A</b> button to see the A's answer. </p>
                <label id="buttonA" class="btn btn-success" onclick="show_qual_answer('buttonA','systemA_answer', 'Include more vegetables on your meal.')"> 
                    System A </label>
                <div id="systemA_answer" class="mt-3"></div>
            </div>
            <br>
            <div id="systemB" class="mt-3"  style="text-align:center">
                <p> And then, Double-Click <b>System B</b> button to see the B's answer. </p>
                <label id="buttonB" class="btn btn-info" onclick='show_qual_answer("buttonB","systemB_answer", "Eating healthy and sustainably can be a challenge, but it doesnt have to be. Eat more vegetables.")'> 
                    System B </label>
                <div id="systemB_answer" class="mt-3"></div>
            </div>
            <br><br>
            <p style="text-align: center">
            <mark>
            <b>Please choose which system's answer aligns and cohere better with the instruction and reference sentences?</b>
            </mark>
            </p>
            <div style="text-align: center">
            <label class="btn btn-success btn-lg">
                <input type="radio" name="instruction_example" value="A"> System A
            </label>
            <label class="btn btn-info btn-lg">
                <input type="radio" name="instruction_example" value="B"> System B
            </label>
            </div>

        </div>
        <br>
        <h4> Explaining the example </h4>
        <p> 
            In the above example, you may first click the system A button and then system B button, accordingly. Then, you may choose either A or B, based on your opinions regarding the coherency with the instruction and reference sentences. For example, if you think the straighforward answer is more coherent, then you may choose A. 
            On the other hand, if you think system B sounds more coherent because of more context, then you may choose B. 
        </p>

        <hr>

        <h2> Next Step </h2> <br>

        <h5> You will be given 30 instruction sets to answer. Please accept this work only if you can make sure to answer all of them thoroughly. Your answer will be verified later for the approval. </h4>
        <p> Next page will prompt you to complete a qualification round to <span style="color:red">check whether you correctly understand the instruction. </span> </p>
        </div>

        <hr>
        <p style='text-align: center'>Click <b><span style="color: blue">Start Qualification</span></b> button to start the qualification round. 
        <b>DO NOT CLICK <span style="background-color:orange">SUBMIT</span> BUTTON!</b>
        <br><br>  
        <button id="qual_button" class="btn btn-primary" onclick="show_qual()">Start Qualification</button>
        </p>

	
	`
	document.getElementById('instruction_page').style.display = "";
	document.getElementById('qual_page').style.display = "none";
	document.getElementById('task_page').style.display = "none";

}

show_instruction();