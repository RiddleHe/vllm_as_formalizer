(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        red - block
        purple - block
        green - block
        orange - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (clear yellow)
        (ontable red)
        (clear red)
        (ontable purple)
        (clear purple)
        (ontable green)
        (clear green)
        (ontable orange)
        (on blue orange)
        (clear blue)
        ; yellow
        ; red
        ; purple
        ; green
        ; orange -> blue
    )
    (:goal (and (on red yellow)
        (on purple red)
        (on orange purple)
        (on green blue)))
)
    