(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        blue - block
        orange - block
        purple - block
        red - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (clear green)
        (ontable blue)
        (clear blue)
        (ontable orange)
        (on purple orange)
        (clear purple)
        (ontable red)
        (clear red)
        (ontable yellow)
        (clear yellow)
        ; green
        ; blue
        ; orange -> purple
        ; red
        ; yellow
    )
    (:goal (and (on orange yellow)
        (on green blue)
        (on red purple)))
)
    