Output-

![image](https://github.com/user-attachments/assets/ff85a21b-0c56-49f1-a3fb-95e2c5de7abc)

### Summary of Findings:

After testing various hyperparameters, here’s what we found:

1. **Learning Rate (\(\alpha\))**:
   - A higher learning rate (e.g., 0.3) allowed for faster learning but caused some instability in reward trends.
   - A moderate learning rate (e.g., 0.1) provided more stable learning over time.

2. **Discount Factor (\(\gamma\))**:
   - A higher discount factor (e.g., 0.9) encouraged the agent to consider future rewards, which helped with planning.
   - A lower discount factor (e.g., 0.8) made the agent focus more on immediate rewards but didn’t lead to the best overall performance.

3. **Exploration Rate (\(\epsilon\))**:
   - A moderate exploration rate (e.g., 0.2) gave a good balance between exploring the environment and exploiting known strategies.

### Optimal Parameters:
The best combination for the agent’s performance was:

- **Learning Rate**: 0.3
- **Discount Factor**: 0.9
- **Exploration Rate**: 0.2

These values resulted in the highest cumulative reward and allowed the agent to navigate the grid effectively.

