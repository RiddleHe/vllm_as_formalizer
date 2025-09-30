(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        orange - block
        blue - block
        green - block
        red - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (clear yellow)
        (ontable orange)
        (clear orange)
        (ontable blue)
        (clear blue)
        (ontable green)
        (clear green)
        (ontable red)
        (clear red)
        (ontable purple)
        (clear purple)
        ; yellow
        ; orange
        ; blue
        ; green
        ; red
        ; purple
    )
    (:goal (and (on red yellow)
        (on blue red)
        (on green blue)
        (on purple green)
        (on orange purple)))
)
    